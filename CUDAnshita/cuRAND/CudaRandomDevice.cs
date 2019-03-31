using System;
using System.Collections.Generic;

namespace CUDAnshita {
	using cudaStream_t = IntPtr;

	/// <summary>
	/// (cuRAND) cuRAND Device API.
	/// </summary>
	public class CudaRandomDevice : IDisposable {
		IntPtr generator = IntPtr.Zero;
		ulong seed = 0;
		List<IntPtr> pointers = new List<IntPtr>();

		public ulong Seed {
			get { return seed; }
			set {
				seed = value;
				cuRAND.SetPseudoRandomGeneratorSeed(generator, value);
			}
		}

		public CudaRandomDevice(curandRngType type = curandRngType.CURAND_RNG_PSEUDO_DEFAULT) {
			generator = cuRAND.CreateGenerator(type);
		}

		~CudaRandomDevice() {
			Dispose();
		}

		public void Dispose() {
			if (generator != IntPtr.Zero) {
				cuRAND.DestroyGenerator(generator);
				generator = IntPtr.Zero;
			}
			foreach (IntPtr pointer in pointers) {
				Runtime.Free(pointer);
			}
			pointers.Clear();
		}

		public IntPtr Generate(int num) {
			IntPtr outputPtr = Malloc(sizeof(uint) * num);
			cuRAND.Generate(generator, outputPtr, num);
			return outputPtr;
		}

		public IntPtr GenerateLogNormal(int num, float mean, float stddev) {
			IntPtr outputPtr = Malloc(sizeof(float) * num);
			cuRAND.GenerateLogNormal(generator, outputPtr, num, mean, stddev);
			return outputPtr;
		}

		public IntPtr GenerateLogNormalDouble(int num, double mean, double stddev) {
			IntPtr outputPtr = Malloc(sizeof(double) * num);
			cuRAND.GenerateLogNormalDouble(generator, outputPtr, num, mean, stddev);
			return outputPtr;
		}

		public IntPtr GenerateLong(int num) {
			IntPtr outputPtr = Malloc(sizeof(ulong) * num);
			cuRAND.GenerateLongLong(generator, outputPtr, num);
			return outputPtr;
		}

		public IntPtr GenerateNormal(int num, float mean, float stddev) {
			IntPtr outputPtr = Malloc(sizeof(float) * num);
			cuRAND.GenerateNormal(generator, outputPtr, num, mean, stddev);
			return outputPtr;
		}

		public IntPtr GenerateNormalDouble(int num, double mean, double stddev) {
			IntPtr outputPtr = Malloc(sizeof(double) * num);
			cuRAND.GenerateNormalDouble(generator, outputPtr, num, mean, stddev);
			return outputPtr;
		}

		public IntPtr GeneratePoisson(int num, double lambda) {
			IntPtr outputPtr = Malloc(sizeof(uint) * num);
			cuRAND.GeneratePoisson(generator, outputPtr, num, lambda);
			return outputPtr;
		}

		public IntPtr GenerateUniform(int num) {
			IntPtr outputPtr = Malloc(sizeof(float) * num);
			cuRAND.GenerateUniform(generator, outputPtr, num);
			return outputPtr;
		}

		public IntPtr GenerateUniformDouble(int num) {
			IntPtr outputPtr = Malloc(sizeof(double) * num);
			cuRAND.GenerateUniformDouble(generator, outputPtr, num);
			return outputPtr;
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

		IntPtr Malloc(long size) {
			IntPtr pointer = Runtime.Malloc(size);
			pointers.Add(pointer);
			return pointer;
		}
	}
}
