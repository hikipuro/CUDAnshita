using System;

namespace CUDAnshita {
	using cudaStream_t = IntPtr;

	/// <summary>
	/// cuRAND random class.
	/// </summary>
	public class Random : IDisposable {
		IntPtr generator = IntPtr.Zero;
		bool isHost = true;
		ulong seed = 0;

		public bool IsHost {
			get { return isHost; }
		}

		public ulong Seed {
			get { return seed; }
			set {
				seed = value;
				cuRAND.SetPseudoRandomGeneratorSeed(generator, value);
			}
		}

		public Random(bool isHost = true, curandRngType type = curandRngType.CURAND_RNG_PSEUDO_DEFAULT) {
			this.isHost = isHost;
			if (isHost) {
				generator = cuRAND.CreateGeneratorHost(type);
			} else {
				generator = cuRAND.CreateGenerator(type);
			}
		}

		~Random() {
			Dispose();
		}

		public void Dispose() {
			if (generator != IntPtr.Zero) {
				cuRAND.DestroyGenerator(generator);
				generator = IntPtr.Zero;
			}
		}

		public uint[] Generate(int num) {
			if (isHost == false) {
				return new uint[0];
			}
			return cuRAND.Generate(generator, num);
		}

		public void Generate(IntPtr outputPtr, int num) {
			if (isHost == true) {
				return;
			}
			cuRAND.Generate(generator, outputPtr, num);
		}

		public float[] GenerateLogNormal(int num, float mean, float stddev) {
			if (isHost == false) {
				return new float[0];
			}
			return cuRAND.GenerateLogNormal(generator, num, mean, stddev);
		}

		public void GenerateLogNormal(IntPtr outputPtr, int num, float mean, float stddev) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateLogNormal(generator, outputPtr, num, mean, stddev);
		}

		public double[] GenerateLogNormalDouble(int num, double mean, double stddev) {
			if (isHost == false) {
				return new double[0];
			}
			return cuRAND.GenerateLogNormalDouble(generator, num, mean, stddev);
		}

		public void GenerateLogNormalDouble(IntPtr outputPtr, int num, double mean, double stddev) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateLogNormalDouble(generator, outputPtr, num, mean, stddev);
		}

		public ulong[] GenerateLong(int num) {
			if (isHost == false) {
				return new ulong[0];
			}
			return cuRAND.GenerateLongLong(generator, num);
		}

		public void GenerateLong(IntPtr outputPtr, int num) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateLongLong(generator, outputPtr, num);
		}

		public float[] GenerateNormal(int num, float mean, float stddev) {
			if (isHost == false) {
				return new float[0];
			}
			return cuRAND.GenerateNormal(generator, num, mean, stddev);
		}

		public void GenerateNormal(IntPtr outputPtr, int num, float mean, float stddev) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateNormal(generator, outputPtr, num, mean, stddev);
		}

		public double[] GenerateNormalDouble(int num, double mean, double stddev) {
			if (isHost == false) {
				return new double[0];
			}
			return cuRAND.GenerateNormalDouble(generator, num, mean, stddev);
		}

		public void GenerateNormalDouble(IntPtr outputPtr, int num, double mean, double stddev) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateNormalDouble(generator, outputPtr, num, mean, stddev);
		}

		public uint[] GeneratePoisson(int num, double lambda) {
			if (isHost == false) {
				return new uint[0];
			}
			return cuRAND.GeneratePoisson(generator, num, lambda);
		}

		public void GeneratePoisson(IntPtr outputPtr, int num, double lambda) {
			if (isHost == true) {
				return;
			}
			cuRAND.GeneratePoisson(generator, outputPtr, num, lambda);
		}

		public float[] GenerateUniform(int num) {
			if (isHost == false) {
				return new float[0];
			}
			return cuRAND.GenerateUniform(generator, num);
		}

		public void GenerateUniform(IntPtr outputPtr, int num) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateUniform(generator, outputPtr, num);
		}

		public double[] GenerateUniformDouble(int num) {
			if (isHost == false) {
				return new double[0];
			}
			return cuRAND.GenerateUniformDouble(generator, num);
		}

		public void GenerateUniformDouble(IntPtr outputPtr, int num) {
			if (isHost == true) {
				return;
			}
			cuRAND.GenerateUniformDouble(generator, outputPtr, num);
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
