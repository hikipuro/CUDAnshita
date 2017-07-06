using System;
using System.Diagnostics;

namespace CUDAnshita_Sample {
	public class Common {
		public static long Benchmark(Action<int> action) {
			GC.Collect();
			Stopwatch sw = Stopwatch.StartNew();
			action(0);
			sw.Stop();
			return sw.ElapsedMilliseconds;
		}

		public static long Benchmark(Action<int> action, int count) {
			GC.Collect();
			Stopwatch sw = Stopwatch.StartNew();
			for (int i = 0; i < count; i++) {
				action(i);
			}
			sw.Stop();
			return sw.ElapsedMilliseconds;
		}
	}
}
