# CUDAnshita
C# から簡単に CUDA を呼び出すためのライブラリ

## 実行に必要なもの
* NVIDIA の GPU と新しめのドライバ
* CUDA Toolkit 10.2
 * nvcuda.dll (ドライバに付属している？)
 * nvrtc64_102_0.dll (CUDA Toolkit に含まれている)
 * cuDNN を使用する場合は、 CUDA Toolkit とは別にダウンロードする必要があります
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
RuntimeCompiler compiler = new RuntimeCompiler();
compiler.AddOptions(
	RuntimeCompiler.OPTION_TARGET_20,
	RuntimeCompiler.OPTION_FMAD_FALSE,
	RuntimeCompiler.OPTION_LINE_INFO
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
```

### トラブルシューティング
* サンプルの実行時にエラーが発生する場合は、64bitビルドで実行してみてください (使用する DLL が 64bit 環境用のものなので、 32bit exe から呼び出すとエラーが発生します)。
* DLLが見つからないエラーが発生する場合は、 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\\[Version]\bin" から、 Visual Studio が exe を出力する bin フォルダにコピーすると解決するかもしれません。

### 注意事項
* cuBLAS, cuFFT, cuFFTW, cuRAND, cuDNN の機能は全くテストできていませんので、CUDAnshita のコードを底の方から修正する気力のあるかたのみ、ご使用ください。
* 旧バージョンの CUDA Toolkit を使用する場合、 CUDAnshita\NVRTC\NVRTC.cs と CUDAnshita\Runtime\Runtime.cs 内の DLL_PATH 変数を書き換えてください。
```cs
// 例: v8.0 を使用する場合
const string DLL_PATH = "nvrtc64_80.dll";
```
* CUDA Toolkit 10.2 以外を使用する場合は cudaError の値が違う場合があります。公式マニュアルを確認してください。

## CUDAnshita の使い方

CUDAnshita の API は 3 つの層に分けて実装されています。
(現段階では未実装箇所が多いため、実装予定という感じです。)

### API Layer 1

DLL の関数を直接呼び出す形式です。

```cs
// 例:
int version = 0;
cudaError error = Runtime.API.cudaRuntimeGetVersion(ref version);
if (error != cudaError.cudaSuccess) {
	Console.WriteLine("Error");
}
```

* クラス名.API.関数名の形式で呼び出す
* DLLImport された関数を直接呼び出す
* 関数呼び出しごとに戻り値のチェックが必要
* C 言語の API に慣れた人向け
* 検索して出てきたコードサンプルを移植する用途向け

### API Layer 2

C# 用の薄いラッパー経由で API を呼び出す形式です。

```cs
// 例:
int version = Runtime.RuntimeGetVersion();
```

* クラス名.関数名の形式で呼び出す
* 関数名は、プリフィックスを取った形式 (cudaRuntimeGetVersion -> RuntimeGetVersion)
* エラー発生時は例外が投げられる (CudaException クラスの例外が発生する)
* CUDA の深い部分まで理解する必要がある人向け
* C 言語の API よりはちょっとだけ楽をしたい人向け

### API Layer 3

C# 用に実装されたクラス経由で API を呼び出す形式です。

* 使用する機能に特化したクラスのインスタンスを作成する
* メモリ確保、解放等の処理は内部で実行される
* CUDA の理解が薄くても問題ないと考える人向け
* 煩雑な処理を書きたくない人向け

## 開発環境
* Visual Studio 2017
* CUDA Toolkit 10.2

## ライセンス
* MIT License

## その他
* APIは今後変更になる可能性があります。

## コードの修正

コードを修正してくださったかたは、プルリクエストを送っていただけると助かります。

主に、次の箇所が気になっています。

* 動かない箇所の修正
  * DLLの関数呼び出しが機能しない
  * ロジックに間違いがある、機能の認識が違ったまま実装されている
* 他のバージョンの CUDA Toolkit への柔軟な実装
* 新機能の追加
* コンパイラが表示する警告の除去
* 単体テストの追加

ここに書かれていない修正でも、送って頂けると嬉しいです。

また、プルリクエストを送るほどでもない軽微な不具合は Issues のところに書いて頂けると修正しやすいです。
Issues には新機能の提案も書いて頂けると参考になります。
(Cuda への理解が足りないところもあり、実装できるかどうかは分かりません)
