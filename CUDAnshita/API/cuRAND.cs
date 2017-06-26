using System;
using System.Runtime.InteropServices;

namespace CUDAnshita.API {
	using size_t = Int64;

	/// <summary>
	/// http://docs.nvidia.com/cuda/curand/
	/// </summary>
	public class cuRAND {
		const string DLL_PATH = "curand64_80.dll";
		const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
		const CharSet CHAR_SET = CharSet.Ansi;

		// ----- Host API

		/// <summary>
		/// Create new random number generator.
		/// </summary>
		/// <param name="generator"></param>
		/// <param name="rng_type"></param>
		/// <returns></returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandCreateGenerator(ref IntPtr generator, curandRngType rng_type);

		/// <summary>
		/// Create new host CPU random number generator.
		/// </summary>
		/// <param name="generator"></param>
		/// <param name="rng_type"></param>
		/// <returns></returns>
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandCreateGeneratorHost(ref IntPtr generator, curandRngType rng_type);

		/*
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* discrete_distribution);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution);
		*/

		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandDestroyGenerator(IntPtr generator);

		/*
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateLogNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateLongLong(curandGenerator_t generator, unsigned long long* outputPtr, size_t num);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGeneratePoisson(curandGenerator_t generator, unsigned int* outputPtr, size_t n, double lambda);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateSeeds(curandGenerator_t generator);
		*/
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateUniform(IntPtr generator, IntPtr outputPtr, size_t num);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGenerateUniformDouble(IntPtr generator, IntPtr outputPtr, size_t num);
		/*
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetDirectionVectors32(curandDirectionVectors32_t* vectors, curandDirectionVectorSet_t set);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetDirectionVectors64(curandDirectionVectors64_t* vectors, curandDirectionVectorSet_t set);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetProperty(libraryPropertyType type, int* value);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetScrambleConstants32(unsigned int** constants);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetScrambleConstants64(unsigned long long** constants);
		*/
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandGetVersion(ref int version);
		/*
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
		*/
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandSetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed);
		/*
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
		
		[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
		static extern curandStatus curandSetStream(curandGenerator_t generator, cudaStream_t stream);
		*/

		// ----- C# Interface

		public static IntPtr CreateGenerator(curandRngType type) {
			IntPtr generator = IntPtr.Zero;
			curandStatus status = curandCreateGenerator(ref generator, type);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
			return generator;
		}

		public static void DestroyGenerator(IntPtr generator) {
			curandStatus status = curandDestroyGenerator(generator);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
		}

		public static IntPtr CreateGeneratorHost(curandRngType type) {
			IntPtr generator = IntPtr.Zero;
			curandStatus status = curandCreateGeneratorHost(ref generator, type);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
			return generator;
		}

		public static void SetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed) {
			curandStatus status = curandSetPseudoRandomGeneratorSeed(generator, seed);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
		}

		public static float[] GenerateUniform(IntPtr generator, int num) {
			float[] result = new float[num];
			int size = sizeof(float) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			curandStatus status = curandGenerateUniform(generator, memory, num);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public static double[] GenerateUniformDouble(IntPtr generator, int num) {
			double[] result = new double[num];
			int size = sizeof(double) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			curandStatus status = curandGenerateUniformDouble(generator, memory, num);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public static int GetVersion() {
			int version = 0;
			curandStatus status = curandGetVersion(ref version);
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
			return version;
		}
	}

	/// <summary>
	/// CURAND choice of direction vector set
	/// </summary>
	public enum curandDirectionVectorSet {
		// Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.
		CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
		// Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.
		CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
		// Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.
		CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
		// Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.
		CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
	}

	/// <summary>
	/// CURAND ordering of results in memory
	/// </summary>
	public enum curandOrdering {
		// Best ordering for pseudorandom results.
		CURAND_ORDERING_PSEUDO_BEST = 100,
		// Specific default 4096 thread sequence for pseudorandom results.
		CURAND_ORDERING_PSEUDO_DEFAULT = 101,
		// Specific seeding pattern for fast lower quality pseudorandom results.
		CURAND_ORDERING_PSEUDO_SEEDED = 102,
		// Specific n-dimensional ordering for quasirandom results.
		CURAND_ORDERING_QUASI_DEFAULT = 201
	}

	/// <summary>
	/// CURAND generator types
	/// </summary>
	public enum curandRngType {
		CURAND_RNG_TEST = 0,
		// Default pseudorandom generator.
		CURAND_RNG_PSEUDO_DEFAULT = 100,
		// XORWOW pseudorandom generator.
		CURAND_RNG_PSEUDO_XORWOW = 101,
		// MRG32k3a pseudorandom generator.
		CURAND_RNG_PSEUDO_MRG32K3A = 121,
		// Mersenne Twister MTGP32 pseudorandom generator.
		CURAND_RNG_PSEUDO_MTGP32 = 141,
		// Mersenne Twister MT19937 pseudorandom generator.
		CURAND_RNG_PSEUDO_MT19937 = 142,
		// PHILOX-4x32-10 pseudorandom generator.
		CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
		// Default quasirandom generator.
		CURAND_RNG_QUASI_DEFAULT = 200,
		// Sobol32 quasirandom generator.
		CURAND_RNG_QUASI_SOBOL32 = 201,
		// Scrambled Sobol32 quasirandom generator.
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
		// Sobol64 quasirandom generator.
		CURAND_RNG_QUASI_SOBOL64 = 203,
		// Scrambled Sobol64 quasirandom generator.
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
	}

	/// <summary>
	/// CURAND function call status types
	/// </summary>
	public enum curandStatus {
		// No errors.
		CURAND_STATUS_SUCCESS = 0,
		// Header file and linked library version do not match.
		CURAND_STATUS_VERSION_MISMATCH = 100,
		// Generator not initialized.
		CURAND_STATUS_NOT_INITIALIZED = 101,
		// Memory allocation failed.
		CURAND_STATUS_ALLOCATION_FAILED = 102,
		// Generator is wrong type.
		CURAND_STATUS_TYPE_ERROR = 103,
		// Argument out of range.
		CURAND_STATUS_OUT_OF_RANGE = 104,
		// Length requested is not a multple of dimension.
		CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
		// GPU does not have double precision required by MRG32k3a.
		CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
		// Kernel launch failure.
		CURAND_STATUS_LAUNCH_FAILURE = 201,
		// Preexisting failure on library entry.
		CURAND_STATUS_PREEXISTING_FAILURE = 202,
		// Initialization of CUDA failed.
		CURAND_STATUS_INITIALIZATION_FAILED = 203,
		// Architecture mismatch, GPU does not support requested feature.
		CURAND_STATUS_ARCH_MISMATCH = 204,
		// Internal library error.
		CURAND_STATUS_INTERNAL_ERROR = 999

	}
}
