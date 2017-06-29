using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using CUDAnshita;
using CUDAnshita.Errors;

namespace CUDAnshita_Sample {
	class cuBLASTest {
		cuBLAS cuBLAS;
		Device device;
		Context context;
		Module module;

		public cuBLASTest() {
			cuBLAS = new cuBLAS();
			//device = new Device(0);
			//context = device.CreateContext();
			//module = new Module();
		}

		~cuBLASTest() {
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
			ExecuteGPU(3);
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
			List<double> c = new List<double>();

			for (int i = 0; i < size * size; i++) {
				//a.Add(i + 1);
				//b.Add((i + 1) * 10);
				//a.Add(1d * i);
				//b.Add(0.1d * i);
				c.Add(0);
			}

			a.AddRange(new double[] { 1, 3, 2, 4 });
			b.AddRange(new double[] { 5, 7, 6, 8 });

			// デバイス上にメモリを転送
			/*DeviceMemory memory = new DeviceMemory();
			memory.Add<float>("a", a);
			memory.Add<float>("b", b);
			memory.Alloc<double>("c", size);
			memory.Alloc<double>("d", size);
			*/

			int elemSize = Marshal.SizeOf(typeof(double));
			int byteSize = elemSize * size * size;
			IntPtr destA = IntPtr.Zero;
			IntPtr destB = IntPtr.Zero;
			IntPtr destC = IntPtr.Zero;
			cudaError result2;
			result2 = CudaRT.cudaMalloc(ref destA, elemSize * 4);
			CudaException.Check(result2, "デバイスメモリの割り当てに失敗しました。");
			result2 = CudaRT.cudaMalloc(ref destB, elemSize * 4);
			CudaException.Check(result2, "デバイスメモリの割り当てに失敗しました。");
			result2 = CudaRT.cudaMalloc(ref destC, byteSize);
			CudaException.Check(result2, "デバイスメモリの割り当てに失敗しました。");

			Console.WriteLine("cuBLAS Test destA: {0}", destA);
			Console.WriteLine("cuBLAS Test destB: {0}", destB);
			Console.WriteLine("cuBLAS Test destC: {0}", destC);

			cuBLAS.SetMatrix(2, 2, a.ToArray(), 2, destA, 2);
			cuBLAS.SetMatrix(2, 2, b.ToArray(), 2, destB, 2);
			cuBLAS.SetMatrix(size, size, c.ToArray(), size, destC, size);
			//float test = cuBLAS.Sdot(2, da, 1, db, 1);
			
			cuBLAS.Dgemm(
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				2, 2, 2,
				1, destA, 2,
				destB, 2,
				0, destC, 2
			);
			/*
			cuBLAS.Dsymm(
				cublasSideMode.CUBLAS_SIDE_RIGHT,
				cublasFillMode.CUBLAS_FILL_MODE_LOWER,
				2, 2,
				1, da, 2,
				db, 2,
				0, dc, 2
			);

			cuBLAS.Dtrmm(
				cublasSideMode.CUBLAS_SIDE_LEFT,
				cublasFillMode.CUBLAS_FILL_MODE_LOWER,
				cublasOperation.CUBLAS_OP_C,
				cublasDiagType.CUBLAS_DIAG_NON_UNIT,
				2, 2,
				1, da, 2,
				db, 2,
				dc, 2
			);
			*/

			double[] rb = cuBLAS.GetMatrixD(2, 2, destB, 2, 2);
			foreach (double cc in rb) {
				Console.WriteLine("cuBLAS Test rb: {0}", cc);
			}

			double[] rc = cuBLAS.GetMatrixD(size, size, destC, size, size);
			foreach (double cc in rc) {
				Console.WriteLine("cuBLAS Test: {0}", cc);
			}

			CudaRT.cudaFree(destA);
			CudaRT.cudaFree(destB);
			CudaRT.cudaFree(destC);

			// 全てのスレッドが終了するまで待つ
			//context.Synchronize();

			// 結果を取得して出力画面に表示
			double[] result = null;// memory.Read<double>("c", size * size);

			// リソースを解放する
			//memory.Dispose();

			return result;
		}

		string Compile(string name, string src) {
			NVRTC compiler = new NVRTC();
			compiler.AddOptions(
				NVRTC.OPTION_TARGET_20,
				NVRTC.OPTION_FMAD_FALSE,
				NVRTC.OPTION_LINE_INFO,
				NVRTC.OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE
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

		int divRoundUp(int value, int radix) {
			return (value + radix - 1) / radix;
		}
	}
}
