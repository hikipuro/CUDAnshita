using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAnshita;

namespace UnitTest {
	/// <summary>
	/// RuntimeTest の概要の説明
	/// </summary>
	[TestClass]
	public class RuntimeTest {
		public RuntimeTest() {
			//
			// TODO: コンストラクター ロジックをここに追加します
			//
		}

		private TestContext testContextInstance;

		/// <summary>
		///現在のテストの実行についての情報および機能を
		///提供するテスト コンテキストを取得または設定します。
		///</summary>
		public TestContext TestContext {
			get {
				return testContextInstance;
			}
			set {
				testContextInstance = value;
			}
		}

		#region 追加のテスト属性
		//
		// テストを作成する際には、次の追加属性を使用できます:
		//
		// クラス内で最初のテストを実行する前に、ClassInitialize を使用してコードを実行してください
		// [ClassInitialize()]
		// public static void MyClassInitialize(TestContext testContext) { }
		//
		// クラス内のテストをすべて実行したら、ClassCleanup を使用してコードを実行してください
		// [ClassCleanup()]
		// public static void MyClassCleanup() { }
		//
		// 各テストを実行する前に、TestInitialize を使用してコードを実行してください
		// [TestInitialize()]
		// public void MyTestInitialize() { }
		//
		// 各テストを実行した後に、TestCleanup を使用してコードを実行してください
		// [TestCleanup()]
		// public void MyTestCleanup() { }
		//
		#endregion

		[TestMethod]
		public void DeviceReset() {
			Runtime.DeviceReset();
		}

		[TestMethod]
		public void GetDevice() {
			int device = Runtime.GetDevice();
			Console.WriteLine(device);
		}

		[TestMethod]
		public void GetDeviceCount() {
			int count = Runtime.GetDeviceCount();
			Console.WriteLine(count);
		}

		[TestMethod]
		public void GetDeviceFlags() {
			uint flags = Runtime.GetDeviceFlags();
			Console.WriteLine(flags);
		}

		[TestMethod]
		public void GetDeviceProperties() {
			cudaDeviceProp prop = Runtime.GetDeviceProperties(0);
			Console.WriteLine(ObjectDumper.Dump(prop));
		}

		[TestMethod]
		public void DriverGetVersion() {
			int version = Runtime.DriverGetVersion();
			Console.WriteLine(version);
		}

		[TestMethod]
		public void RuntimeGetVersion() {
			int version = Runtime.RuntimeGetVersion();
			Console.WriteLine(version);
		}
	}
}
