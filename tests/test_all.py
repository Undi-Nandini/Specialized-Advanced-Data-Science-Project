"""
tests/test_all.py
Full test suite — 30 tests across pipeline, network, engine, watcher modules.
Uses pytest-style parametrize annotations (also runnable via unittest).
"""

import sys
import json
import pickle
import tempfile
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.text_pipeline import PipelineConfig, TextCleaner, Vocabulary
from network.sentiment_net  import ReviewSentimentNet, GRUCell, LinearLayer, EmbedLayer
from watcher.monitor        import RollingMonitor, _percentile, _kl_divergence


# ── helper: tiny config ───────────────────────────────────────────────────────
def _cfg(**kw) -> PipelineConfig:
    defaults = dict(max_vocab=50, max_seq_len=8, lowercase=True,
                    strip_urls=True, strip_html=True, drop_stopwords=False)
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _mini_model() -> ReviewSentimentNet:
    return ReviewSentimentNet(vocab_size=50, embed_dim=8,
                              gru1_dim=6, gru2_dim=4,
                              hidden_dim=6, num_classes=3)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TextCleaner
# ══════════════════════════════════════════════════════════════════════════════

class TestTextCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = TextCleaner(_cfg())

    # 1. URLs removed
    def test_url_removed(self):
        out = self.cleaner.transform("Visit http://shop.example.com today")
        self.assertNotIn("http", out)
        self.assertIn("visit", out)

    # 2. HTML stripped
    def test_html_stripped(self):
        out = self.cleaner.transform("<strong>Amazing</strong> product")
        self.assertNotIn("<strong>", out)
        self.assertIn("amazing", out)

    # 3. Lowercase applied
    def test_lowercase(self):
        out = self.cleaner.transform("GREAT QUALITY")
        self.assertEqual(out, "great quality")

    # 4. Special chars removed
    def test_special_chars(self):
        out = self.cleaner.transform("wow!!! #great @user")
        for ch in "!#@":
            self.assertNotIn(ch, out)

    # 5. Empty string returns empty
    def test_empty_returns_empty(self):
        self.assertEqual(self.cleaner.transform(""), "")

    # 6. None returns empty
    def test_none_returns_empty(self):
        self.assertEqual(self.cleaner.transform(None), "")

    # 7. Whitespace collapsed
    def test_whitespace_collapsed(self):
        out = self.cleaner.transform("hello    world   test")
        self.assertNotIn("  ", out)

    # 8. Stopword removal (config enabled)
    def test_stopword_removal(self):
        c = TextCleaner(_cfg(drop_stopwords=True))
        out = c.transform("this is a great product")
        self.assertNotIn(" a ", f" {out} ")

    # 9. DataFrame processing adds columns
    def test_df_processing(self):
        import pandas as pd
        df = pd.DataFrame({"text": ["Great item!", "Terrible quality"], "label": [2, 0]})
        out = self.cleaner.fit_transform_df(df, col="text")
        self.assertIn("clean_text",  out.columns)
        self.assertIn("word_count",  out.columns)
        self.assertEqual(len(out), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.cfg   = _cfg(max_vocab=30)
        self.vocab = Vocabulary(self.cfg)
        corpus     = ["great product quality", "bad awful terrible",
                      "okay decent average", "wonderful brilliant superb"]
        self.vocab.build(corpus)

    # 10. PAD at index 0
    def test_pad_index_zero(self):
        self.assertEqual(self.vocab.w2i["<PAD>"], 0)

    # 11. UNK at index 1
    def test_unk_index_one(self):
        self.assertEqual(self.vocab.w2i["<UNK>"], 1)

    # 12. Known word encoded
    def test_known_word_encoded(self):
        seq = self.vocab.encode(["great product"])
        self.assertTrue(np.any(seq[0] > 1))

    # 13. Unknown word maps to UNK (1)
    def test_unknown_word_is_unk(self):
        seq = self.vocab.encode(["xyzzy florp"])
        self.assertTrue(np.all(seq[0][seq[0] > 0] == 1))

    # 14. Output shape matches max_seq_len
    def test_output_shape(self):
        seqs = self.vocab.encode(["great", "okay product", "terrible awful bad"])
        self.assertEqual(seqs.shape[1], self.cfg.max_seq_len)

    # 15. Short sequences are zero-padded
    def test_short_padded(self):
        seqs = self.vocab.encode(["hi"])
        self.assertTrue(np.any(seqs[0] == 0))

    # 16. Batch encoding correct shape
    def test_batch_shape(self):
        seqs = self.vocab.encode(["a", "b c", "d e f"])
        self.assertEqual(seqs.shape[0], 3)

    # 17. Save and reload gives identical encoding
    def test_save_reload_identical(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        self.vocab.save(path)
        v2   = Vocabulary.load(path)
        seq1 = self.vocab.encode(["great product"])
        seq2 = v2.encode(["great product"])
        np.testing.assert_array_equal(seq1, seq2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ReviewSentimentNet
# ══════════════════════════════════════════════════════════════════════════════

class TestReviewSentimentNet(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.model = _mini_model()

    # 18. Forward output shape
    def test_forward_shape(self):
        x = np.random.randint(0, 50, (5, 8))
        p = self.model.forward(x)
        self.assertEqual(p.shape, (5, 3))

    # 19. Probabilities sum to 1
    def test_probs_sum_one(self):
        x = np.random.randint(0, 50, (4, 8))
        p = self.model.forward(x)
        np.testing.assert_allclose(p.sum(axis=-1), np.ones(4), atol=1e-5)

    # 20. All probabilities non-negative
    def test_probs_non_negative(self):
        x = np.random.randint(0, 50, (4, 8))
        p = self.model.forward(x)
        self.assertTrue(np.all(p >= 0))

    # 21. predict returns valid class indices
    def test_predict_classes_valid(self):
        x = np.random.randint(0, 50, (6, 8))
        classes, _ = self.model.predict(x)
        self.assertTrue(np.all((classes >= 0) & (classes <= 2)))

    # 22. Save/load preserves forward output
    def test_save_load_consistency(self):
        x = np.random.randint(0, 50, (3, 8))
        orig = self.model.forward(x)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        self.model.save(path)
        m2 = ReviewSentimentNet.load(path)
        np.testing.assert_allclose(orig, m2.forward(x), atol=1e-5)

    # 23. param_count positive
    def test_param_count_positive(self):
        self.assertGreater(self.model.param_count(), 0)

    # 24. Repeated identical inputs give same output
    def test_deterministic(self):
        x = np.random.randint(0, 50, (1, 8))
        p1 = self.model.forward(x)
        p2 = self.model.forward(x)
        np.testing.assert_array_equal(p1, p2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRUCell
# ══════════════════════════════════════════════════════════════════════════════

class TestGRUCell(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.gru = GRUCell(in_dim=8, h_dim=6)

    # 25. forward shape without sequence
    def test_forward_shape(self):
        x = np.random.randn(4, 5, 8).astype(np.float32)
        h = self.gru.forward(x, return_seq=False)
        self.assertEqual(h.shape, (4, 6))

    # 26. forward shape with sequence
    def test_forward_seq_shape(self):
        x = np.random.randn(4, 5, 8).astype(np.float32)
        h = self.gru.forward(x, return_seq=True)
        self.assertEqual(h.shape, (4, 5, 6))


# ══════════════════════════════════════════════════════════════════════════════
# 5. RollingMonitor
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingMonitor(unittest.TestCase):

    def setUp(self):
        self.mon = RollingMonitor(window=100, log_dir="/tmp/test_mon_logs")

    # 27. Record increments counter
    def test_record_increments(self):
        self.mon.record("Positive", 0.9, 30.0)
        self.mon.record("Negative", 0.8, 25.0)
        self.assertEqual(self.mon._all_n, 2)

    # 28. Error flag tracked
    def test_error_tracked(self):
        self.mon.record("Neutral", 0.5, 50.0, is_error=True)
        s = self.mon.compute()
        self.assertEqual(s.n_errors, 1)
        self.assertGreater(s.error_rate, 0)

    # 29. No-drift on uniform distribution
    def test_no_drift_uniform(self):
        for _ in range(90):
            for lbl in ["Negative", "Neutral", "Positive"]:
                self.mon.record(lbl, 0.8, 30.0)
        s = self.mon.compute()
        self.assertFalse(s.drift_alert)

    # 30. Latency percentiles ordered correctly
    def test_latency_percentiles_ordered(self):
        for lat in range(10, 110, 5):
            self.mon.record("Positive", 0.8, float(lat))
        s = self.mon.compute()
        self.assertLessEqual(s.lat_p50, s.lat_p95)
        self.assertLessEqual(s.lat_p95, s.lat_p99)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestTextCleaner, TestVocabulary, TestReviewSentimentNet,
                TestGRUCell, TestRollingMonitor]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'━'*50}")
    print(f"  Tests run   : {result.testsRun}")
    print(f"  Failures    : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    status = "✅ ALL PASSED" if result.wasSuccessful() else "❌ SOME FAILED"
    print(f"  Status      : {status}")
    print(f"{'━'*50}\n")
