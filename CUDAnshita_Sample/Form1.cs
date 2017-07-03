using CUDAnshita;
using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace CUDAnshita_Sample {
	/// <summary>
	/// https://developer.nvidia.com/cuda-gpus
	/// </summary>
	public partial class Form1 : Form {
		// CUDA のプログラム (cu 形式)
		string addKernelString = @"
extern ""C"" __global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
";

		public Form1() {
			InitializeComponent();
			Runtime.DeviceReset();
		}

		private void buttonTest_Click(object sender, EventArgs e) {
			TestCompile();
			//TestCudaRT();
			//TestCuRAND();
			//TestCuBLAS();
			//TestMatrix();
			//TestCuDNN();
		}

		private void TestCompile() {
			// プログラムのコンパイル (cu から PTX へ)
			RuntimeCompiler compiler = new RuntimeCompiler();
			//compiler.AddHeader("curand_kernel.h", @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\curand_kernel.h");
			compiler.AddOptions(
				RuntimeCompiler.OPTION_TARGET_20,
				RuntimeCompiler.OPTION_FMAD_FALSE,
				RuntimeCompiler.OPTION_LINE_INFO,
				RuntimeCompiler.OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE
				//RuntimeCompiler.OPTION_INCLUDE_PATH_ + @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
			);

			string ptx = compiler.Compile("addKernel.cu", addKernelString);

			if (ptx == null) {
				Console.WriteLine("Compile Error:");
				Console.WriteLine();
				Console.WriteLine(compiler.Log);
				return;
			}

			// コンパイル時のログを出力画面に表示
			Console.WriteLine("----- <Compile Log>");
			Console.WriteLine(compiler.Log);
			Console.WriteLine("----- </Compile Log>");

			// コンパイル済みプログラムを出力画面に表示
			Console.WriteLine("----- <PTX>");
			Console.WriteLine(ptx);
			Console.WriteLine("----- </PTX>");

			// プログラムの実行準備
			Device device = new Device(0);
			Context context = device.CreateContext();
			Module module = new Module();

			Console.WriteLine(device.Name);
			Console.WriteLine(device.PCIBusId);
			Console.WriteLine(device.TotalMem);
			//Console.WriteLine(device.GetProperties());
			Console.WriteLine(context.ApiVersion);
			//return;

			// PTX データをロード
			module.LoadData(ptx);

			// いったんメインメモリ上に変数を準備
			const int arraySize = 5;
			List<int> a = new List<int>();
			List<int> b = new List<int>();

			for (int i = 0; i < arraySize; i++) {
				a.Add(i + 1);
				b.Add((i + 1) * 10);
			}

			// デバイス上にメモリを転送
			DeviceMemory memory = new DeviceMemory();
			memory.Add<int>("a", a);
			memory.Add<int>("b", b);
			memory.Alloc<int>("c", arraySize);

			// 関数の実行
			module.SetBlockCount(1, 1, 1);
			module.SetThreadCount(arraySize, 1, 1);
			module.Excecute(
				"addKernel",
				memory["c"],
				memory["a"],
				memory["b"]
			);

			// 全てのスレッドが終了するまで待つ
			context.Synchronize();

			// 結果を取得して出力画面に表示
			int[] results = memory.Read<int>("c", arraySize);
			Console.WriteLine("----- <Execute Log>");
			for (int i = 0; i < arraySize; i++) {
				Console.WriteLine("{0} + {1}  = {2}", a[i], b[i], results[i]);
			}
			Console.WriteLine("----- </Execute Log>");

			// リソースを解放する
			memory.Dispose();
			module.Dispose();
			context.Dispose();
			device.Dispose();
		}

		private void TestCudaRT() {
			int count = 0;
			Runtime.API.cudaGetDeviceCount(ref count);
			Console.WriteLine("CudaRT: {0}", count);
		}

		private void TestCuRAND() {
			CudaRandomHost rand = new CudaRandomHost();
			rand.Seed = (ulong)DateTime.Now.Ticks;
			//uint[] test = rand.Generate(10);
			//float[] test = rand.GenerateUniform(10);
			double[] test = rand.GenerateUniformDouble(10);
			foreach (var i in test) {
				Console.WriteLine("cuRAND: {0}", i);
			}
		}

		cuBLASTest cuBLASTest;
		private void TestCuBLAS() {
			Console.WriteLine("cuBLAS: {0}", cuBLAS.GetVersion());
			if (cuBLASTest == null) {
				cuBLASTest = new cuBLASTest();
			}
			cuBLASTest.Test();
		}

		private void TestMatrix() {
			MatrixTest matrixTest = new MatrixTest();
			matrixTest.Test();
			matrixTest.Dispose();
		}

		private void TestCuDNN() {
			cuDNN6 cudnn = new cuDNN6();
			Console.WriteLine("cudnnGetVersion: {0}", cuDNN6.GetVersion());
			Console.WriteLine("cudnnGetCudartVersion: {0}", cuDNN6.GetCudartVersion());
			Console.WriteLine("cudnnGetErrorString: {0}", cuDNN6.GetErrorString(cudnnStatus.CUDNN_STATUS_SUCCESS));

			var config = Runtime.DeviceGetCacheConfig();
			Console.WriteLine(config);
			Console.WriteLine(Runtime.DeviceGetPCIBusId(0));

			Console.WriteLine(Driver.DeviceGetPCIBusId(0));

			int totalDevices = Runtime.GetDeviceCount();

			for (int i = 0; i < totalDevices; i++) {
				cudaDeviceProp prop = Runtime.GetDeviceProperties(i);

				Console.WriteLine("device {0}, {1}", i, prop.name);
				Console.WriteLine("sms {0}", prop.multiProcessorCount);
				Console.WriteLine("Capabilities {0}.{1}", prop.major, prop.minor);
				Console.WriteLine("SmClock {0} Mhz", (float)prop.clockRate * 1e-3);
				Console.WriteLine("MemSize (Mb) {0}", (int)(prop.totalGlobalMem / (1024 * 1024)));
				Console.WriteLine("MemClock {0} Mhz", (float)prop.memoryClockRate * 1e-3);
			}

			//DNNTest test = new DNNTest();
			//test.Test();
		}
	}
}
