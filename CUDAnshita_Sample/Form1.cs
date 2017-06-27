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
		}

		private void buttonTest_Click(object sender, EventArgs e) {
			//TestCompile();
			//TestCudaRT();
			//TestCuRAND();
			TestMatrix();
		}

		private void TestCompile() {
			// プログラムのコンパイル (cu から PTX へ)
			NVRTC compiler = new NVRTC();
			//compiler.AddHeader("curand_kernel.h", @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\curand_kernel.h");
			compiler.AddOptions(
				NVRTC.OPTION_TARGET_20,
				NVRTC.OPTION_FMAD_FALSE,
				NVRTC.OPTION_LINE_INFO,
				NVRTC.OPTION_DEVICE_AS_DEFAULT_EXECUTION_SPACE
				//NVRTC.OPTION_INCLUDE_PATH_ + @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
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
			CudaRT.cudaGetDeviceCount(ref count);
			Console.WriteLine("CudaRT: {0}", count);
		}

		private void TestCuRAND() {
			cuRAND rand = new cuRAND();
			rand.Seed = 1234;
			int[] test = rand.Generate(10);
			foreach (int i in test) {
				Console.WriteLine("cuRAND: {0}", i);
			}
		}

		MatrixTest matrixTest;
		private void TestMatrix() {
			if (matrixTest == null) {
				matrixTest = new MatrixTest();
			}
			matrixTest.Test();
		}
	}
}
