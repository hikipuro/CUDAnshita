# CUDAnshita
C#から簡単にCUDAを呼び出すためのライブラリ

## 実行に必要なもの
* NVIDIA の GPU と新しめのドライバ
* CUDA Toolkit 8.0
 * nvcuda.dll (ドライバに付属している？)
 * nvrtc64_80.dll (CUDA Toolkit 8.0に含まれている)
* .NET Framework 2.0 以降

## コードサンプル
### cu形式のプログラムの準備
```cs
// C#の文字列
string addKernelString = @"
extern ""C"" __global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
";
```

### usingの設定
```cs
using CUDAnshita;
```

### cuのコンパイル
```cs
// プログラムのコンパイル (cu から PTX へ)
NVRTC compiler = new NVRTC();
compiler.AddOptions(
	NVRTC.OPTION_TARGET_20,
	NVRTC.OPTION_FMAD_FALSE,
	NVRTC.OPTION_LINE_INFO
);
string ptx = compiler.Compile(
	"addKernel.cu",
	addKernelString
);

if (ptx == null) {
	Console.WriteLine("Compile Error:");
	Console.WriteLine();
	Console.WriteLine(compiler.Log);
	return;
}

// コンパイル時のログを出力画面に表示
Console.WriteLine(compiler.Log);

// コンパイル済みプログラムを出力画面に表示
Console.WriteLine(ptx);
```

### PTXの実行
```cs
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

for (var i = 0; i < arraySize; i++) {
	a.Add(i + 1);        // 1, 2, 3, 4, 5
	b.Add((i + 1) * 10); // 10, 20, 30, 40, 50
}

// デバイス上にメモリを転送
DeviceMemory memory = new DeviceMemory();
memory.Add<int>("a", a); // メインメモリからデバイスメモリへ転送
memory.Add<int>("b", b);
memory.Alloc<int>("c", arraySize); // デバイスメモリの確保

// 関数の実行
module.SetBlockCount(1, 1, 1);
module.SetThreadCount(arraySize, 1, 1);
module.Excecute(
	"addKernel", // 関数名
	memory["c"], // 引数 0
	memory["a"], // 引数 1
	memory["b"]  // 引数 2
);

// 全てのスレッドが終了するまで待つ
context.Synchronize();

// 結果を取得して出力画面に表示
int[] results = memory.Read<int>("c", arraySize);
for (int i = 0; i < arraySize; i++) {
	Console.WriteLine("{0} + {1}  = {2}", a[i], b[i], results[i]);
}

// リソースを解放する
memory.Dispose();
module.Dispose();
context.Dispose();
device.Dispose();
```

### トラブルシューティング
* サンプルの実行時にエラーが発生する場合は、64bitビルドで実行してみてください (使用する DLL が 64bit 環境用のものなので、 32bit exe から呼び出すとエラーが発生します)。
* DLLが見つからないエラーが発生する場合は、 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin" から、 Visual Studio が exe を出力する bin フォルダにコピーすると解決するかもしれません。

### 注意事項
* cuBLAS, cuFFT, cuFFTW, cuRAND, cuDNN の機能は全くテストできていませんので、CUDAnshita のコードを底の方から修正する気力のあるかたのみ、ご使用ください。

## 開発環境
* Visual Studio 2015
* CUDA Toolkit 8.0

## ライセンス
* MIT License

## その他
* APIは今後変更になる可能性があります。
