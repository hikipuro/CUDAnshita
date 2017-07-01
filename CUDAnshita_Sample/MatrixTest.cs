using System;
using System.Collections.Generic;
using System.Diagnostics;
using CUDAnshita;

namespace CUDAnshita_Sample {
	class MatrixTest {
		string matrixProgram = @"
extern ""C"" {
	__global__ void matrixAdd(double *a, double *b, double *c, int width, int height) {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x >= width || y >= height) {
			return;
		}
		int index = y * width + x;
		c[index] = a[index] + b[index];
	}
	__global__ void matrixDot(double *a, double *b, double *c, int width, int height) {
		int x = blockDim.x * blockIdx.x + threadIdx.x;
		int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x >= width || y >= height) {
			return;
		}
		double total = 0;
		for (int i = 0; i < width; i++) {
			double r = a[y * width + i] * b[i * width + x];
			total += r;
		}
		int index = y * width + x;
		c[index] = total;
	}
}
";
		string ptx;
		Device device;
		Context context;
		Module module;

		public MatrixTest() {
			device = new Device(0);
			context = device.CreateContext();
			module = new Module();
		}

		~MatrixTest() {
			if (module != null) {
				module.Dispose();
				module = null;
			}
			if (context != null) {
				context.Dispose();
				context = null;
			}
			if (device != null) {
				device.Dispose();
				device = null;
			}
		}

		public void Test() {
			int loopCount = 2;
			int size = 300;

			long timeCpu = Benchmark((t) => {
				for (int i = 0; i < loopCount; i++) {
					ExecuteCPU(size);
				}
			});

			if (ptx == null) {
				ptx = Compile("matrixAdd.cu", matrixProgram);
				module = new Module();
				module.LoadData(ptx);
			}
			long timeGpu = Benchmark((t) => {
				for (int i = 0; i < loopCount; i++) {
					ExecuteGPU(size);
				}
			});

			Console.WriteLine("time CPU: {0}", timeCpu);
			Console.WriteLine("time GPU: {0}", timeGpu);

			// 値のチェック
			double[] cpuResult = ExecuteCPU(size);
			double[] gpuResult = ExecuteGPU(size);
			for (int i = 0; i < size * size; i++) {
				//Console.WriteLine("result[{0}]: {1}", i, cpuResult[i]);
				if (cpuResult[i] != gpuResult[i]) {
					throw new Exception();
				}
			}
		}

		long Benchmark(Action<int> action) {
			GC.Collect();
			Stopwatch sw = Stopwatch.StartNew();
			action(0);
			sw.Stop();
			return sw.ElapsedMilliseconds;
		}

		double[] ExecuteCPU(int size) {
			Matrix2D matrix1 = new Matrix2D(size, size);
			Matrix2D matrix2 = new Matrix2D(size, size);
			for (var i = 0; i < size * size; i++) {
				matrix1[i] = i + 1;
				matrix2[i] = (i + 1) * 10;
			}
			matrix1 = matrix1.Dot(matrix2);
			return matrix1.Data;
		}

		double[] ExecuteGPU(int size) {
			// いったんメインメモリ上に変数を準備
			List<double> a = new List<double>();
			List<double> b = new List<double>();

			for (int i = 0; i < size * size; i++) {
				a.Add(i + 1);
				b.Add((i + 1) * 10);
			}

			// デバイス上にメモリを転送
			DeviceMemory memory = new DeviceMemory();
			memory.Add<double>("a", a);
			memory.Add<double>("b", b);
			memory.Alloc<double>("c", size * size);

			// 関数の実行
			CallMethod(
				"matrixDot", size, size,
				memory["a"],
				memory["b"],
				memory["c"],
				size,
				size
			);

			// 全てのスレッドが終了するまで待つ
			context.Synchronize();

			// 結果を取得して出力画面に表示
			double[] result = memory.Read<double>("c", size * size);

			// リソースを解放する
			memory.Dispose();

			return result;
		}

		string Compile(string name, string src) {
			RuntimeCompiler compiler = new RuntimeCompiler();
			compiler.AddOptions(
				RuntimeCompiler.OPTION_TARGET_20,
				RuntimeCompiler.OPTION_FMAD_FALSE,
				RuntimeCompiler.OPTION_LINE_INFO,
				RuntimeCompiler.OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE
			);

			string ptx = compiler.Compile(name, src);
			if (ptx == null) {
				Console.WriteLine("Compile Error:");
				Console.WriteLine();
				Console.WriteLine(compiler.Log);
				return null;
			}
			return ptx;
		}

		void CallMethod(string name, int width, int height, params object[] args) {
			int threadX = 64;
			int threadY = 2;
			int blockX = divRoundUp(width, threadX);
			int blockY = divRoundUp(height, threadY);

			module.SetBlockCount(blockX, blockY, 1);
			module.SetThreadCount(threadX, threadY, 1);
			module.Excecute(name, args);
		}

		int divRoundUp(int value, int radix) {
			return (value + radix - 1) / radix;
		}
	}
}
