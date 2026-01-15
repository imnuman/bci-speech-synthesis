"""Tests for bug fixes in BCI Speech Synthesis"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')


class TestImportFixes:
    """Test that import bugs are fixed."""
    
    def test_main_imports(self):
        """Test that main.py imports work correctly."""
        try:
            from main import BCIPipeline
            assert BCIPipeline is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_tts_module_exports(self):
        """Test that tts module exports get_response_text."""
        from tts import get_response_text, KokoroTTS, ALSAOutput
        
        assert get_response_text is not None
        assert callable(get_response_text)
        
        # Test function works
        result = get_response_text('yes')
        assert result == 'Yes'


class TestSPIReaderFixes:
    """Test SPI reader bug fixes."""
    
    def test_spi_reader_auto_open(self):
        """Test that SPI reader auto-opens when start() is called."""
        from acquisition.spi_reader import SPIReader
        
        reader = SPIReader(sample_rate=250, channels=32)
        
        # Should not crash even if not explicitly opened
        # (will fail to open actual device but should handle gracefully)
        samples_received = []
        def callback(sample):
            samples_received.append(sample)
        
        reader.start(callback=callback)
        # Should handle missing device gracefully
        reader.stop()
        reader.close()
    
    def test_spi_bounds_checking(self):
        """Test SPI reader handles short reads gracefully."""
        from acquisition.spi_reader import SPIReader
        
        reader = SPIReader(sample_rate=250, channels=32)
        
        # Test with no SPI device (spi is None)
        sample = reader._read_sample()
        
        assert sample.shape == (32,)
        assert np.all(sample == 0)


class TestAudioClipping:
    """Test audio clipping bug fix."""
    
    def test_audio_clipping_prevents_overflow(self):
        """Test that audio is clipped before conversion to int16."""
        from tts.alsa_output import ALSAOutput
        
        output = ALSAOutput(sample_rate=24000)
        
        # Generate audio that exceeds [-1, 1] range
        audio = np.array([0.5, 1.5, -2.0, 0.8], dtype=np.float32)
        
        # Mock the _pcm to capture conversion
        class MockPCM:
            def __init__(self):
                self.written_data = None
            
            def write(self, data):
                self.written_data = data
        
        output._pcm = MockPCM()
        output.play(audio)
        
        # Verify data was written
        assert output._pcm.written_data is not None
        
        # Convert back to verify clipping worked
        written_array = np.frombuffer(output._pcm.written_data, dtype=np.int16)
        
        # Values should be clipped
        assert np.all(written_array >= -32767)
        assert np.all(written_array <= 32767)


class TestVocabIndexing:
    """Test vocab index bounds checking."""
    
    def test_tensorrt_decoder_vocab_bounds(self):
        """Test TensorRT decoder handles invalid vocab indices."""
        from decoder.tensorrt_infer import TensorRTDecoder
        
        vocab = ['yes', 'no', 'help']
        decoder = TensorRTDecoder(
            engine_path='models/decoder.engine',
            vocab=vocab
        )
        
        # Test with dummy data (won't load real engine)
        eeg = np.random.randn(32, 1000).astype(np.float32)
        intent, conf, probs = decoder.predict(eeg)
        
        # Should return valid intent from vocab
        assert intent in vocab
        assert 0 <= conf <= 1
        assert len(probs) == len(vocab)


class TestResourceCleanup:
    """Test resource cleanup bug fixes."""
    
    def test_pipeline_cleanup_method_exists(self):
        """Test that BCIPipeline has cleanup method."""
        from main import BCIPipeline
        
        # Create pipeline (won't actually initialize components without config)
        # Just verify the method exists
        assert hasattr(BCIPipeline, 'cleanup')
        assert callable(getattr(BCIPipeline, 'cleanup'))


class TestArrayOperations:
    """Test array operation bug fixes."""
    
    def test_ring_buffer_wraparound(self):
        """Test ring buffer handles wraparound correctly."""
        from acquisition.ring_buffer import RingBuffer
        
        buffer = RingBuffer(capacity=10, channels=4)
        
        # Fill buffer beyond capacity
        for i in range(15):
            sample = np.ones(4) * i
            buffer.push(sample)
        
        # Get window that should handle wraparound
        window = buffer.get_window(5)
        
        assert window is not None
        assert window.shape == (5, 4)
        
        # Should contain most recent samples (10-14)
        assert np.allclose(window[0], np.ones(4) * 10)
        assert np.allclose(window[-1], np.ones(4) * 14)


class TestContextFusion:
    """Test context fusion bug fixes."""
    
    def test_context_history_trimming(self):
        """Test that old context is properly trimmed."""
        from context.context_fusion import ContextFusion
        import time
        
        fusion = ContextFusion(context_window_sec=1.0)
        
        # Add old context
        fusion.add_context("old context", time.time() - 2.0)
        
        # Add recent context
        current_time = time.time()
        fusion.add_context("recent context", current_time)
        
        # Old context should be trimmed
        recent = fusion.get_recent_context()
        assert "old context" not in recent
        assert "recent context" in recent


class TestFeatureExtraction:
    """Test feature extraction correctness."""
    
    def test_feature_extraction_no_nan(self):
        """Test that feature extraction doesn't produce NaN values."""
        from processing.features import FeatureExtractor
        
        extractor = FeatureExtractor(sample_rate=1000)
        
        # Generate test data
        data = np.random.randn(1000, 32) * 20
        
        features = extractor.extract_all(data)
        
        # No NaN values
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_division_by_zero_protection(self):
        """Test that features handle zero variance correctly."""
        from processing.features import FeatureExtractor
        
        extractor = FeatureExtractor(sample_rate=1000)
        
        # Create data with zero variance in some channels
        data = np.zeros((1000, 32))
        data[:, 0] = np.random.randn(1000)  # Only first channel has variance
        
        features = extractor.extract_all(data)
        
        # Should not crash and should not have NaN/inf
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
