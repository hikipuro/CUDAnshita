using System;

namespace CUDAnshita {
	using cudaStream_t = IntPtr;

	/// <summary>
	/// cuRAND Host API.
	/// </summary>
	public class CudaRandomHost : IDisposable {
		IntPtr generator = IntPtr.Zero;
		ulong seed = 0;

		public ulong Seed {
			get { return seed; }
			set {
				seed = value;
				cuRAND.SetPseudoRandomGeneratorSeed(generator, value);
			}
		}

		public CudaRandomHost(curandRngType type = curandRngType.CURAND_RNG_PSEUDO_DEFAULT) {
			generator = cuRAND.CreateGeneratorHost(type);
		}

		~CudaRandomHost() {
			Dispose();
		}

		public void Dispose() {
			if (generator != IntPtr.Zero) {
				cuRAND.DestroyGenerator(generator);
				generator = IntPtr.Zero;
			}
		}

		public uint[] Generate(int num) {
			return cuRAND.Generate(generator, num);
		}

		public float[] GenerateLogNormal(int num, float mean, float stddev) {
			return cuRAND.GenerateLogNormal(generator, num, mean, stddev);
		}

		public double[] GenerateLogNormalDouble(int num, double mean, double stddev) {
			return cuRAND.GenerateLogNormalDouble(generator, num, mean, stddev);
		}

		public ulong[] GenerateLong(int num) {
			return cuRAND.GenerateLongLong(generator, num);
		}

		public float[] GenerateNormal(int num, float mean, float stddev) {
			return cuRAND.GenerateNormal(generator, num, mean, stddev);
		}

		public double[] GenerateNormalDouble(int num, double mean, double stddev) {
			return cuRAND.GenerateNormalDouble(generator, num, mean, stddev);
		}

		public uint[] GeneratePoisson(int num, double lambda) {
			return cuRAND.GeneratePoisson(generator, num, lambda);
		}

		public float[] GenerateUniform(int num) {
			return cuRAND.GenerateUniform(generator, num);
		}

		public double[] GenerateUniformDouble(int num) {
			return cuRAND.GenerateUniformDouble(generator, num);
		}

		public void SetGeneratorOffset(ulong offset) {
			cuRAND.SetGeneratorOffset(generator, offset);
		}

		public void SetGeneratorOrdering(curandOrdering order) {
			cuRAND.SetGeneratorOrdering(generator, order);
		}

		public void SetQuasiRandomGeneratorDimensions(uint num_dimensions) {
			cuRAND.SetQuasiRandomGeneratorDimensions(generator, num_dimensions);
		}

		public void SetStream(cudaStream_t stream) {
			cuRAND.SetStream(generator, stream);
		}
	}
}
