"""Offline integration checks; no provider or LangSmith requests."""
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['OTEL_SDK_DISABLED'] = 'true'
import sys
import json
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.embeddings import DeterministicFakeEmbedding
from utils.data_loader import build_vectorstore
rag = importlib.import_module('01_langsmith_rag_pipeline')
ab = importlib.import_module('02_prompt_hub_ab_routing')
evaluation = importlib.import_module('03_ragas_evaluation')
validators = importlib.import_module('04_guardrails_validator')
from guardrails import Guard

class LabChecks(unittest.TestCase):
    def test_rag_and_dataset(self):
        store = build_vectorstore(['FAISS indexes vectors.', 'RAG retrieves context.', 'LangSmith traces calls.'], DeterministicFakeEmbedding(size=32))
        llm = FakeListChatModel(responses=['FAISS indexes vectors.'])
        with patch.object(rag, 'get_llm', return_value=llm):
            chain, retriever = rag.build_rag_chain(store)
        self.assertEqual(rag.ask(chain, 'What is FAISS?'), 'FAISS indexes vectors.')
        out = evaluation.run_rag(retriever, llm, ab.PROMPT_V1, 'What is FAISS?')
        self.assertEqual(len(out['contexts']), 3)
        sample = {'question': 'What is FAISS?', 'reference': 'FAISS indexes vectors.', **out}
        self.assertEqual(len(evaluation.build_ragas_dataset([sample]).samples), 1)
        routed = ab.ask_ab(retriever, llm, ab.PROMPT_V2, sample['question'], 'v2')
        self.assertEqual(routed['version'], 'v2')

    def test_step1_main_runs_all_questions(self):
        with patch.object(rag.config, 'validate', return_value=True), \
             patch.object(rag, 'setup_vectorstore', return_value=object()), \
             patch.object(rag, 'build_rag_chain', return_value=(object(), object())), \
             patch.object(rag, 'ask', return_value='answer') as ask, \
             patch('builtins.print'):
            rag.main()
        self.assertEqual(ask.call_count, 50)

    def test_evaluation_retries_only_missing_score_and_resumes(self):
        import tempfile
        from types import SimpleNamespace
        sample = {"question": "Q", "reference": "R", "answer": "A", "contexts": ["C"]}
        model = SimpleNamespace(model_dump=lambda: {"model": "offline"})
        initial = {"faithfulness": [0.0], "answer_relevancy": [0.8],
                   "context_recall": [float("nan")], "context_precision": [1.0]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            with patch.object(evaluation, '__file__', str(root / 'src' / 'evaluation.py')), \
                 patch.object(evaluation, 'get_llm', return_value=model), \
                 patch.object(evaluation, 'get_embeddings', return_value=model), \
                 patch.object(evaluation, 'evaluate', side_effect=[initial, {"context_recall": [0.75]}]) as evaluate:
                scores = evaluation.run_ragas_eval([sample], 'test')
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual([m.name for m in evaluate.call_args.kwargs['metrics']], ['context_recall'])
                self.assertEqual(scores['faithfulness'], 0.0)
                self.assertEqual(scores['context_recall'], 0.75)
                evaluate.reset_mock()
                self.assertEqual(evaluation.run_ragas_eval([sample], 'test'), scores)
                evaluate.assert_not_called()

    def test_router_and_hub_fallback(self):
        versions = [ab.get_prompt_version(f'req-{i:04d}') for i in range(50)]
        self.assertEqual(set(versions), {ab.PROMPT_V1_NAME, ab.PROMPT_V2_NAME})
        self.assertEqual(versions, [ab.get_prompt_version(f'req-{i:04d}') for i in range(50)])
        class OfflineClient:
            def pull_prompt(self, name):
                raise ConnectionError('offline')
        self.assertEqual(ab.pull_prompts_from_hub(OfflineClient())[ab.PROMPT_V1_NAME], ab.PROMPT_V1)

    def test_pii_guard(self):
        guard = Guard().use(validators.PIIDetector(on_fail=validators.OnFailAction.FIX))
        for text, tag in [('x@example.com','EMAIL'), ('(555) 867-5309','PHONE'), ('123-45-6789','SSN'), ('4532 1234 5678 9010','CREDIT_CARD')]:
            with self.subTest(tag=tag):
                output = guard.validate(text).validated_output
                self.assertNotIn(text, output)
                self.assertIn(f'[{tag}_REDACTED]', output)
        self.assertEqual(guard.validate('clean text').validated_output, 'clean text')
        output = guard.validate('x@example.com 555-123-4567').validated_output
        self.assertIn('[EMAIL_REDACTED]', output)
        self.assertIn('[PHONE_REDACTED]', output)

    def test_json_guard(self):
        guard = Guard().use(validators.JSONFormatter(on_fail=validators.OnFailAction.FIX))
        cases = [('{"x":1}', {'x':1}), ('```json\n{"x":1}\n```', {'x':1}), ("{'x':1}", {'x':1}), ('{"x":1,}', {'x':1}), ('broken {]', {'error':'invalid_json'}), ('{"text":"Bob\'s",}', {'text':"Bob's"})]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(json.loads(guard.validate(text).validated_output), expected)

if __name__ == '__main__':
    unittest.main()
