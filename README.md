# CUDAnshita
C#から簡単にCUDAを呼び出すためのライブラリ

## 実行に必要なもの
* nVidiaのGPUと新しめのドライバ
* CUDA Toolkit 7.5
 * nvcuda.dll (ドライバに付属している？)
 * nvrtc64_75.dll (CUDA Toolkit 7.5に含まれている)

## コードサンプル
### cu形式のプログラムの準備
```
// C#の文字列
string addKernelString = @"
extern ""C"" __global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
";
```

### usingの設定
```
using CUDAnshita;
using CUDAnshita.API;
```

### cuのコンパイル
```
// プログラムのコンパイル (cu から PTX へ)
NVRTC.Program program = new NVRTC.Program();
program.Create(addKernelString, "addKernel.cu", null, null);
program.Compile(
	NVRTC.Program.OPTION_TARGET_20, 
	NVRTC.Program.OPTION_FMAD_FALSE,
	NVRTC.Program.OPTION_LINE_INFO
);

// コンパイル時のログを出力画面に表示
Console.WriteLine(program.GetLog());

// コンパイル済みプログラムを取得 (PTX 形式)
string ptx = program.GetPTX();

// コンパイル済みプログラムを出力画面に表示
Console.WriteLine(ptx);

// コンパイラの後処理
program.Dispose();
```

### PTXの実行
```
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

## 開発環境
* Visual Studio 2013
* CUDA Toolkit 7.5

## ライセンス
* MIT License

## その他
* APIは今後変更になる可能性があります。
