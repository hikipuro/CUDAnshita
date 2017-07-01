using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAnshita;
using System.Collections.Generic;

namespace UnitTest {
	[TestClass]
	public class cuRANDTest {
		[TestMethod]
		public void CreateGenerator() {
			IntPtr generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_MRG32K3A);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			try {
				generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_MT19937);
				Assert.AreNotEqual(IntPtr.Zero, generator);
				cuRAND.DestroyGenerator(generator);
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_ARCH_MISMATCH.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_MTGP32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_PHILOX4_32_10);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_XORWOW);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_QUASI_DEFAULT);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_QUASI_SOBOL32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_QUASI_SOBOL64);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_TEST);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void CreateGeneratorHost() {
			IntPtr generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_MRG32K3A);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_MT19937);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_MTGP32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_PHILOX4_32_10);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_XORWOW);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_DEFAULT);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL32);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL64);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_TEST);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void DestroyGenerator() {
			IntPtr generator = IntPtr.Zero;
			try {
				cuRAND.DestroyGenerator(generator);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = IntPtr.Zero;
			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			Assert.AreNotEqual(IntPtr.Zero, generator);
			cuRAND.DestroyGenerator(generator);
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void CreatePoissonDistribution() {
			IntPtr distribution = IntPtr.Zero;
			distribution = cuRAND.CreatePoissonDistribution(1);
			Assert.AreNotEqual(IntPtr.Zero, distribution);
			cuRAND.DestroyDistribution(distribution);
		}

		[TestMethod]
		public void DestroyDistribution() {
			IntPtr distribution = IntPtr.Zero;
			try {
				cuRAND.DestroyDistribution(distribution);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}
		}

		[TestMethod]
		public void SetPseudoRandomGeneratorSeed() {
			ulong seed = 1234;
			int num = 10;
			IntPtr generator = IntPtr.Zero;

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			cuRAND.SetPseudoRandomGeneratorSeed(generator, seed);
			uint[] array1 = cuRAND.Generate(generator, num);
			uint[] array1_2 = cuRAND.Generate(generator, num);
			cuRAND.DestroyGenerator(generator);

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			cuRAND.SetPseudoRandomGeneratorSeed(generator, seed);
			uint[] array2 = cuRAND.Generate(generator, num);
			cuRAND.DestroyGenerator(generator);

			//DebugPrintArray(array1);
			//DebugPrintArray(array2);
			Assert.AreEqual(num, array1.Length);
			Assert.AreEqual(num, array1_2.Length);
			Assert.AreEqual(num, array2.Length);

			Assert.IsTrue(IsSameArray(array1, array2));
			Assert.IsFalse(IsSameArray(array1, array1_2));

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			cuRAND.SetPseudoRandomGeneratorSeed(generator, 0);
			uint[] array3 = cuRAND.Generate(generator, num);
			cuRAND.DestroyGenerator(generator);

			Assert.AreEqual(num, array3.Length);
			Assert.IsFalse(IsSameArray(array1, array3));
		}

		[TestMethod]
		public void Generate() {
			IntPtr generator = IntPtr.Zero;
			uint[] array = null;
			try {
				array = cuRAND.Generate(generator, 1);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
			array = cuRAND.Generate(generator, 1);
			Assert.AreEqual(1, array.Length);
			array = cuRAND.Generate(generator, 0);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.Generate(generator, -1);
			Assert.AreNotEqual(null, array);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.Generate(generator, 10);
			Assert.AreEqual(10, array.Length);
			Assert.AreNotEqual(10, CountZeroItems(array));
			cuRAND.DestroyGenerator(generator);

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL64);
			try {
				array = cuRAND.Generate(generator, 1);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_TYPE_ERROR.ToString();
				Assert.AreEqual(message, e.Message);
			}
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void GenerateLogNormal() {
			IntPtr generator = IntPtr.Zero;
			float[] array = null;
			try {
				array = cuRAND.GenerateLogNormal(generator, 1, 0, 0);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

			float mean = 0.5f;
			float stddev = 0.5f;

			array = cuRAND.GenerateLogNormal(generator, 2, mean, stddev);
			Assert.AreEqual(2, array.Length);
			array = cuRAND.GenerateLogNormal(generator, 0, mean, stddev);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLogNormal(generator, -1, mean, stddev);
			Assert.AreNotEqual(null, array);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLogNormal(generator, 10, mean, stddev);
			Assert.AreEqual(10, array.Length);
			Assert.AreNotEqual(10, CountZeroItems(array));
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void GenerateLogNormalDouble() {
			IntPtr generator = IntPtr.Zero;
			double[] array = null;

			try {
				array = cuRAND.GenerateLogNormalDouble(generator, 1, 0.5, 0.5);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL64);

			double mean = 0.1;
			double stddev = 1;

			array = cuRAND.GenerateLogNormalDouble(generator, 2, mean, stddev);
			Assert.AreEqual(2, array.Length);
			array = cuRAND.GenerateLogNormalDouble(generator, 0, mean, stddev);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLogNormalDouble(generator, -1, mean, stddev);
			Assert.AreNotEqual(null, array);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLogNormalDouble(generator, 10, mean, stddev);
			Assert.AreEqual(10, array.Length);
			Assert.AreNotEqual(10, CountZeroItems(array));
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void GenerateLongLong() {
			IntPtr generator = IntPtr.Zero;
			ulong[] array = null;

			try {
				array = cuRAND.GenerateLongLong(generator, 1);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL64);

			array = cuRAND.GenerateLongLong(generator, 2);
			Assert.AreEqual(2, array.Length);
			array = cuRAND.GenerateLongLong(generator, 0);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLongLong(generator, -1);
			Assert.AreNotEqual(null, array);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateLongLong(generator, 10);
			Assert.AreEqual(10, array.Length);
			Assert.AreNotEqual(10, CountZeroItems(array));
			cuRAND.DestroyGenerator(generator);
		}

		[TestMethod]
		public void GenerateNormal() {
			IntPtr generator = IntPtr.Zero;
			float[] array = null;

			try {
				array = cuRAND.GenerateNormal(generator, 1, 0.5f, 0.5f);
				Assert.Fail();
			} catch (CudaException e) {
				string message = curandStatus.CURAND_STATUS_NOT_INITIALIZED.ToString();
				Assert.AreEqual(message, e.Message);
			}

			generator = cuRAND.CreateGeneratorHost(curandRngType.CURAND_RNG_QUASI_SOBOL64);

			float mean = 0.1f;
			float stddev = 1f;

			array = cuRAND.GenerateNormal(generator, 2, mean, stddev);
			Assert.AreEqual(2, array.Length);
			array = cuRAND.GenerateNormal(generator, 0, mean, stddev);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateNormal(generator, -1, mean, stddev);
			Assert.AreNotEqual(null, array);
			Assert.AreEqual(0, array.Length);
			array = cuRAND.GenerateNormal(generator, 10, mean, stddev);
			Assert.AreEqual(10, array.Length);
			Assert.AreNotEqual(10, CountZeroItems(array));
			cuRAND.DestroyGenerator(generator);
		}

		private int CountZeroItems<T>(T[] array) {
			if (array == null) {
				return 0;
			}
			int total = 0;
			foreach (T item in array) {
				if (item.Equals(0)) {
					total++;
				}
			}
			return total;
		}

		private bool IsSameArray<T>(T[] array1, T[] array2) {
			if (array1 == null || array2 == null) {
				return false;
			}
			if (array1.Length != array2.Length) {
				return false;
			}

			int length = array1.Length;
			for (int i = 0; i < length; i++) {
				if (array1[i].Equals(array2[i]) == false) {
					return false;
				}
			}
			return true;
		}

		private void DebugPrintArray<T>(T[] array) {
			int length = array.Length;
			for (int i = 0; i < length; i++) {
				Console.WriteLine("{0}: {1}", i, array[i]);
			}
		}
	}
}
