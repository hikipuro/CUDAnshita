using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAnshita;

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

			//generator = IntPtr.Zero;
			//generator = cuRAND.CreateGenerator(curandRngType.CURAND_RNG_PSEUDO_MT19937);
			//Assert.AreNotEqual(IntPtr.Zero, generator);
			//cuRAND.DestroyGenerator(generator);

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
	}
}
