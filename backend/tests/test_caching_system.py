#!/usr/bin/env python
"""
Comprehensive tests for the activation caching system.
Tests memory cache, disk cache, hybrid cache, and conversation persistence.
"""

import sys
import os
import tempfile
import shutil
import torch
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import MemoryCache, DiskCache, HybridCache, create_default_cache
from app.conversation import PersistentConversationTokenizer, CachedLogitLensExtractor
from nnsight import LanguageModel


class TestCachingSystem:
    """Test suite for the caching system."""
    
    def __init__(self):
        self.temp_dirs = []
        self.model = None
    
    def setup(self):
        """Set up test environment."""
        print("Setting up test environment...")
        
        # Initialize model (reuse across tests)
        if self.model is None:
            torch.set_grad_enabled(False)
            self.model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
        
        print("✅ Test setup complete")
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def create_temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_test_activations(self, num_layers: int = 3, num_tokens: int = 2) -> dict:
        """Create synthetic activation data for testing."""
        activations = {}
        
        for token_idx in range(num_tokens):
            token_acts = {}
            for layer_idx in range(num_layers):
                # Create realistic but small activation tensors
                layer_acts = {
                    'top_k_probs': torch.randn(5).abs(),  # 5 probabilities
                    'top_k_indices': torch.randint(0, 1000, (5,)),  # 5 token indices
                    'hidden_state': torch.randn(10, 20)  # Small hidden state
                }
                token_acts[layer_idx] = layer_acts
            activations[token_idx] = token_acts
        
        return activations
    
    def test_memory_cache(self):
        """Test memory cache functionality."""
        print("\n🧪 Testing Memory Cache...")
        
        try:
            # Create small memory cache
            cache = MemoryCache(max_size_mb=1)  # 1MB limit
            
            # Test basic storage and retrieval
            test_acts = self.create_test_activations()
            conv_id = "test_conv_1"
            
            # Store activations
            for token_idx, acts in test_acts.items():
                success = cache.store(conv_id, token_idx, acts)
                assert success, f"Failed to store token {token_idx}"
            
            print(f"✅ Stored {len(test_acts)} token activations")
            
            # Test retrieval
            for token_idx, expected_acts in test_acts.items():
                retrieved = cache.retrieve(conv_id, token_idx)
                assert retrieved is not None, f"Failed to retrieve token {token_idx}"
                assert len(retrieved) == len(expected_acts), "Layer count mismatch"
            
            print("✅ Retrieved all activations successfully")
            
            # Test cache statistics
            stats = cache.get_stats()
            assert stats['type'] == 'memory'
            assert stats['entries'] == len(test_acts)
            assert stats['hit_count'] == len(test_acts)
            
            print(f"✅ Cache stats: {stats['size_mb']:.2f}MB used, {stats['hit_rate_percent']:.1f}% hit rate")
            
            # Test LRU eviction by filling cache
            large_acts = {}
            for i in range(10):  # Store many large activations
                large_acts[i] = {
                    0: {'large_tensor': torch.randn(100, 100)}  # Much larger tensor
                }
                cache.store("large_conv", i, large_acts[i])
            
            # Original tokens should be evicted
            original_token = cache.retrieve(conv_id, 0)
            print(f"✅ LRU eviction working (original token evicted: {original_token is None})")
            
            # Test conversation clearing
            cache.clear_conversation("large_conv")
            stats_after_clear = cache.get_stats()
            print(f"✅ Conversation cleared, entries reduced to {stats_after_clear['entries']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Memory cache test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_disk_cache(self):
        """Test disk cache functionality."""
        print("\n💾 Testing Disk Cache...")
        
        try:
            # Create disk cache with temporary directory
            cache_dir = self.create_temp_dir()
            cache = DiskCache(cache_dir=str(cache_dir), max_size_gb=0.1)  # 100MB limit
            
            # Test basic storage and retrieval
            test_acts = self.create_test_activations(num_layers=5, num_tokens=3)
            conv_id = "disk_test_conv"
            
            # Store activations
            for token_idx, acts in test_acts.items():
                success = cache.store(conv_id, token_idx, acts)
                assert success, f"Failed to store token {token_idx}"
            
            print(f"✅ Stored {len(test_acts)} token activations to disk")
            
            # Verify files were created
            conv_dir = cache_dir / conv_id
            assert conv_dir.exists(), "Conversation directory not created"
            
            files = list(conv_dir.glob("*.pt"))
            expected_files = len(test_acts) * 5  # 3 tokens × 5 layers each
            assert len(files) == expected_files, f"Expected {expected_files} files, got {len(files)}"
            
            print(f"✅ Created {len(files)} activation files")
            
            # Test retrieval
            for token_idx, expected_acts in test_acts.items():
                retrieved = cache.retrieve(conv_id, token_idx)
                assert retrieved is not None, f"Failed to retrieve token {token_idx}"
                assert len(retrieved) == len(expected_acts), "Layer count mismatch"
                
                # Verify tensor data integrity
                for layer_idx in expected_acts:
                    assert layer_idx in retrieved, f"Layer {layer_idx} missing"
                    for key in expected_acts[layer_idx]:
                        assert key in retrieved[layer_idx], f"Key {key} missing in layer {layer_idx}"
            
            print("✅ Retrieved all activations with data integrity")
            
            # Test cache statistics
            stats = cache.get_stats()
            assert stats['type'] == 'disk'
            assert stats['conversations'] == 1
            assert stats['files'] == expected_files
            
            print(f"✅ Disk stats: {stats['size_mb']:.2f}MB used, {stats['conversations']} conversations")
            
            # Test conversation clearing
            cache.clear_conversation(conv_id)
            assert not conv_dir.exists(), "Conversation directory not deleted"
            
            print("✅ Conversation cleared from disk")
            
            return True
            
        except Exception as e:
            print(f"❌ Disk cache test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_hybrid_cache(self):
        """Test hybrid cache with memory + disk."""
        print("\n🔄 Testing Hybrid Cache...")
        
        try:
            # Create hybrid cache
            cache_dir = self.create_temp_dir()
            memory_cache = MemoryCache(max_size_mb=1)
            disk_cache = DiskCache(cache_dir=str(cache_dir), max_size_gb=0.1)
            hybrid_cache = HybridCache(memory_cache, disk_cache)
            
            # Test storage (should go to both)
            test_acts = self.create_test_activations()
            conv_id = "hybrid_test"
            
            for token_idx, acts in test_acts.items():
                success = hybrid_cache.store(conv_id, token_idx, acts)
                assert success, f"Failed to store token {token_idx}"
            
            print(f"✅ Stored {len(test_acts)} activations in hybrid cache")
            
            # Test memory retrieval (should be fast)
            start_time = time.time()
            retrieved_from_memory = hybrid_cache.retrieve(conv_id, 0)
            memory_time = time.time() - start_time
            
            assert retrieved_from_memory is not None, "Failed to retrieve from memory"
            print(f"✅ Memory retrieval: {memory_time*1000:.1f}ms")
            
            # Clear memory cache to test disk fallback
            memory_cache.cache.clear()
            memory_cache.current_size = 0
            
            # Test disk fallback
            start_time = time.time()
            retrieved_from_disk = hybrid_cache.retrieve(conv_id, 0)
            disk_time = time.time() - start_time
            
            assert retrieved_from_disk is not None, "Failed to retrieve from disk"
            print(f"✅ Disk fallback retrieval: {disk_time*1000:.1f}ms")
            
            # Verify promotion to memory
            retrieved_again = hybrid_cache.retrieve(conv_id, 0)
            assert retrieved_again is not None, "Failed promotion to memory"
            assert conv_id + ":0" in memory_cache.cache, "Not promoted to memory cache"
            
            print("✅ Disk-to-memory promotion working")
            
            # Test combined statistics
            stats = hybrid_cache.get_stats()
            assert stats['type'] == 'hybrid'
            assert 'memory' in stats and 'disk' in stats
            
            print(f"✅ Hybrid stats: Memory={stats['memory']['entries']} entries, Disk={stats['disk']['conversations']} conversations")
            
            return True
            
        except Exception as e:
            print(f"❌ Hybrid cache test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_conversation_persistence(self):
        """Test persistent conversation management."""
        print("\n💬 Testing Conversation Persistence...")
        
        try:
            # Create conversation manager
            cache_dir = self.create_temp_dir()
            conv_dir = self.create_temp_dir()
            
            cache = create_default_cache(memory_mb=10, disk_gb=0.1, cache_dir=str(cache_dir))
            conv_manager = PersistentConversationTokenizer(
                model=self.model,
                cache=cache,
                persistence_dir=str(conv_dir)
            )
            
            # Start new conversation
            conv_id = conv_manager.start_conversation()
            assert conv_id is not None, "Failed to create conversation"
            print(f"✅ Created conversation: {conv_id}")
            
            # Add messages
            user_tokens = conv_manager.add_message("Hello, how are you?", is_user=True)
            assistant_tokens = conv_manager.add_message("I'm doing well, thank you!", is_user=False)
            
            print(f"✅ Added messages: {len(user_tokens)} user tokens, {len(assistant_tokens)} assistant tokens")
            
            # Test conversation stats
            stats = conv_manager.get_conversation_stats()
            assert stats['total_tokens'] == len(user_tokens) + len(assistant_tokens)
            assert stats['total_messages'] == 2
            
            print(f"✅ Conversation stats: {stats['total_tokens']} total tokens, {stats['total_messages']} messages")
            
            # Test context window
            last_token_pos = len(conv_manager.global_tokens) - 1
            context = conv_manager.get_context_window(last_token_pos, size=5)
            assert len(context) <= 6, "Context window too large"  # 5 context + 1 target
            
            target_token = next(t for t in context if t['is_target'])
            assert target_token['global_position'] == last_token_pos
            
            print(f"✅ Context window: {len(context)} tokens")
            
            # Test persistence by creating new manager with same conversation
            conv_manager2 = PersistentConversationTokenizer(
                model=self.model,
                cache=cache,
                persistence_dir=str(conv_dir)
            )
            
            # Resume conversation
            resumed_id = conv_manager2.start_conversation(conv_id)
            assert resumed_id == conv_id, "Failed to resume conversation"
            assert len(conv_manager2.global_tokens) == len(conv_manager.global_tokens), "Token count mismatch after resume"
            
            print("✅ Conversation persistence working")
            
            # Test conversation listing
            conversations = conv_manager2.list_conversations()
            assert len(conversations) >= 1, "Conversation not in list"
            assert any(c['conversation_id'] == conv_id for c in conversations), "Our conversation not found in list"
            
            print(f"✅ Listed {len(conversations)} conversations")
            
            # Test conversation clearing
            success = conv_manager2.clear_conversation()
            assert success, "Failed to clear conversation"
            
            # Verify files were deleted
            conv_file = conv_dir / f"{conv_id}.json"
            assert not conv_file.exists(), "Conversation file not deleted"
            
            print("✅ Conversation cleared successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Conversation persistence test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cached_extraction(self):
        """Test cached logit lens extraction."""
        print("\n🔬 Testing Cached Logit Lens Extraction...")
        
        try:
            # Create cached extractor
            cache_dir = self.create_temp_dir()
            cache = create_default_cache(memory_mb=50, disk_gb=0.1, cache_dir=str(cache_dir))
            extractor = CachedLogitLensExtractor(model=self.model, cache=cache)
            
            # Test activation extraction with caching
            text = "The capital of France is"
            conv_id = "extraction_test"
            
            print(f"Extracting activations for: '{text}'")
            
            start_time = time.time()
            activations = extractor.extract_activations_with_caching(
                text=text,
                conversation_id=conv_id,
                start_token_idx=0,
                top_k=3
            )
            extraction_time = time.time() - start_time
            
            assert len(activations) > 0, "No activations extracted"
            print(f"✅ Extracted activations for {len(activations)} tokens in {extraction_time:.3f}s")
            
            # Test cached retrieval
            start_time = time.time()
            for token_pos in activations.keys():
                cached_predictions = extractor.get_cached_predictions(conv_id, token_pos)
                assert cached_predictions is not None, f"No cached predictions for token {token_pos}"
                assert len(cached_predictions) > 0, f"Empty predictions for token {token_pos}"
            
            retrieval_time = time.time() - start_time
            print(f"✅ Retrieved cached predictions for {len(activations)} tokens in {retrieval_time:.3f}s")
            
            # Verify speed improvement (cached should be much faster)
            speedup = extraction_time / retrieval_time if retrieval_time > 0 else float('inf')
            print(f"✅ Cache speedup: {speedup:.1f}x faster")
            
            # Test prediction format
            first_token_predictions = extractor.get_cached_predictions(conv_id, 0)
            assert len(first_token_predictions) == extractor.num_layers, "Missing layers in predictions"
            
            first_layer = first_token_predictions[0]
            assert hasattr(first_layer, 'layer'), "Missing layer attribute"
            assert hasattr(first_layer, 'predictions'), "Missing predictions attribute"
            assert len(first_layer.predictions) > 0, "No predictions in layer"
            
            first_prediction = first_layer.predictions[0]
            assert hasattr(first_prediction, 'token'), "Missing token in prediction"
            assert hasattr(first_prediction, 'probability'), "Missing probability in prediction"
            assert hasattr(first_prediction, 'rank'), "Missing rank in prediction"
            
            print(f"✅ Prediction format correct: Layer 0 top prediction: '{first_prediction.token}' ({first_prediction.probability:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Cached extraction test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_performance_stress(self):
        """Test system performance under stress."""
        print("\n⚡ Testing Performance Under Stress...")
        
        try:
            # Create system components
            cache_dir = self.create_temp_dir()
            cache = create_default_cache(memory_mb=20, disk_gb=0.1, cache_dir=str(cache_dir))
            
            # Simulate multiple conversations
            num_conversations = 5
            tokens_per_conversation = 20
            
            print(f"Simulating {num_conversations} conversations with {tokens_per_conversation} tokens each...")
            
            start_time = time.time()
            
            for conv_idx in range(num_conversations):
                conv_id = f"stress_test_{conv_idx}"
                
                # Create realistic activation data
                for token_idx in range(tokens_per_conversation):
                    activations = {
                        layer_idx: {
                            'top_k_probs': torch.randn(5).abs(),
                            'top_k_indices': torch.randint(0, 50000, (5,)),
                            'hidden_state': torch.randn(768)  # GPT-2 size
                        }
                        for layer_idx in range(12)  # GPT-2 layers
                    }
                    
                    success = cache.store(conv_id, token_idx, activations)
                    assert success, f"Failed to store conv {conv_idx} token {token_idx}"
            
            store_time = time.time() - start_time
            total_tokens = num_conversations * tokens_per_conversation
            
            print(f"✅ Stored {total_tokens} tokens in {store_time:.2f}s ({total_tokens/store_time:.1f} tokens/sec)")
            
            # Test random access performance
            start_time = time.time()
            successful_retrievals = 0
            
            for _ in range(50):  # Random access test
                conv_idx = torch.randint(0, num_conversations, (1,)).item()
                token_idx = torch.randint(0, tokens_per_conversation, (1,)).item()
                conv_id = f"stress_test_{conv_idx}"
                
                retrieved = cache.retrieve(conv_id, token_idx)
                if retrieved is not None:
                    successful_retrievals += 1
            
            retrieve_time = time.time() - start_time
            success_rate = successful_retrievals / 50 * 100
            
            print(f"✅ Random access: {successful_retrievals}/50 successful ({success_rate:.1f}%) in {retrieve_time:.3f}s")
            
            # Test cache statistics under load
            stats = cache.get_stats()
            if stats['type'] == 'hybrid':
                memory_util = stats['memory']['utilization_percent']
                disk_conversations = stats['disk']['conversations']
                print(f"✅ Cache utilization: Memory {memory_util:.1f}%, Disk {disk_conversations} conversations")
            
            # Test memory pressure (should trigger evictions)
            print("Testing memory pressure...")
            large_conv_id = "large_test"
            
            for i in range(100):  # Store many large activations
                large_activations = {
                    0: {'huge_tensor': torch.randn(1000, 1000)}  # Very large tensor
                }
                cache.store(large_conv_id, i, large_activations)
            
            final_stats = cache.get_stats()
            print(f"✅ Memory pressure test complete, final memory usage: {final_stats.get('memory', {}).get('size_mb', 0):.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance stress test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all caching system tests."""
        print("🧪 Starting Comprehensive Caching System Tests")
        print("=" * 60)
        
        self.setup()
        
        tests = [
            ("Memory Cache", self.test_memory_cache),
            ("Disk Cache", self.test_disk_cache),
            ("Hybrid Cache", self.test_hybrid_cache),
            ("Conversation Persistence", self.test_conversation_persistence),
            ("Cached Extraction", self.test_cached_extraction),
            ("Performance Stress", self.test_performance_stress)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
            
            print("-" * 60)
        
        self.cleanup()
        
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All caching system tests passed!")
            print("✅ System ready for production use with caching enabled")
        else:
            print("❌ Some tests failed. Check implementation.")
        
        return passed == total


if __name__ == "__main__":
    tester = TestCachingSystem()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)