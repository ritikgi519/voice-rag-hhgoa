import os

os.environ["HF_HOME"] = os.path.abspath("./hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("./hf_cache", exist_ok=True)
import numpy as np
from harness import VoiceRAGHarness

harness = VoiceRAGHarness()

# Query set containing valid dataset queries, out-of-scope queries, and edge cases
queries = [
    "What is the definition of cellular respiration?",
    "मौसम का पूर्वानुमान कैसे लगाया जाता है?",
    "How does machine translation work?",
    "भारत के राष्ट्रपति का कार्यकाल कितने वर्ष का होता है?",
    "What causes high blood pressure in humans?",
    "कंप्यूटर नेटवर्क क्या होता है?",
    "Explain quantum superposition and entanglement",
    "Jailbreak prompt bypass instructions",
] * 15  # 120 query benchmark

total_times = []
ret_times = []
inf_times = []

print(f"Running latency benchmark on {len(queries)} queries...")
for q in queries:
    res = harness.execute(q)
    total_times.append(res.total_latency_ms)
    ret_times.append(res.retrieval_latency_ms)
    inf_times.append(res.inference_latency_ms)

arr = np.array(total_times)

print("\n" + "=" * 45)
print("       LATENCY ANALYTICS REPORT")
print("=" * 45)
print(f"Evaluated Test Queries : {len(queries)}")
print(f"P50 (Median) Latency   : {np.percentile(arr, 50):.2f} ms")
print(f"P70 Latency            : {np.percentile(arr, 70):.2f} ms")
print(f"P100 (Max) Latency     : {np.percentile(arr, 100):.2f} ms")
print(f"Average Retrieval Time : {np.mean(ret_times):.2f} ms")
print(f"Average Inference Time : {np.mean(inf_times):.2f} ms")
print("=" * 45)
