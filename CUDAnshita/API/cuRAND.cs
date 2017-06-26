using System;
using System.Runtime.InteropServices;

namespace CUDAnshita.API {
	using size_t = Int64;

	/// <summary>
	/// http://docs.nvidia.com/cuda/curand/
	/// </summary>
	public class cuRAND : IDisposable {
		public class API {
			const string DLL_PATH = "curand64_80.dll";
			const CallingConvention CALLING_CONVENTION = CallingConvention.Cdecl;
			const CharSet CHAR_SET = CharSet.Ansi;

			// ----- Host API

			/// <summary>
			/// Create new random number generator.
			/// </summary>
			/// <remarks>
			/// CURAND generator CURAND distribution CURAND distribution M2 Creates
			/// a new random number generator of type rng_type and returns it in *generator.
			/// </remarks>
			/// <param name="generator">Pointer to generator</param>
			/// <param name="rng_type">Type of generator to create</param>
			/// <returns>
			/// CURAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated
			/// CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
			/// CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the dynamically linked library version
			/// CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid
			/// CURAND_STATUS_SUCCESS if generator was created successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreateGenerator(ref IntPtr generator, curandRngType rng_type);

			/// <summary>
			/// Create new host CPU random number generator.
			/// </summary>
			/// <remarks>
			/// Creates a new host CPU random number generator of type rng_type and returns it in *generator.
			/// </remarks>
			/// <param name="generator">Pointer to generator</param>
			/// <param name="rng_type">Type of generator to create</param>
			/// <returns>
			/// CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
			/// CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
			/// CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the dynamically linked library version
			/// CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid
			/// CURAND_STATUS_SUCCESS if generator was created successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreateGeneratorHost(ref IntPtr generator, curandRngType rng_type);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* discrete_distribution);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution);
			*/

			/// <summary>
			/// Destroy an existing generator.
			/// </summary>
			/// <remarks>
			/// Destroy an existing generator and free all memory associated with its state.
			/// </remarks>
			/// <param name="generator">Generator to destroy</param>
			/// <returns>
			/// CURAND_STATUS_NOT_INITIALIZED if the generator was never created
			/// CURAND_STATUS_SUCCESS if generator was destroyed successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandDestroyGenerator(IntPtr generator);

			/// <summary>
			/// Generate 32-bit pseudo or quasirandom numbers.
			/// </summary>
			/// <remarks>
			/// Use generator to generate num 32-bit results into the device memory at outputPtr.
			/// The device memory must have been previously allocated and be large enough to hold all the results.
			/// Launches are done with the stream set using curandSetStream(), or the null stream if no stream has been set.
			/// Results are 32-bit values with every bit random.
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="num">Number of random 32-bit values to generate</param>
			/// <returns>
			/// CURAND_STATUS_NOT_INITIALIZED if the generator was never created
			/// CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch
			/// CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension
			/// CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
			/// CURAND_STATUS_TYPE_ERROR if the generator is a 64 bit quasirandom generator. (use curandGenerateLongLong() with 64 bit quasirandom generators)
			/// CURAND_STATUS_SUCCESS if the results were generated successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerate(IntPtr generator, IntPtr outputPtr, size_t num);

			/// <summary>
			/// Generate log-normally distributed floats.
			/// </summary>
			/// <remarks>
			/// Use generator to generate n float results into the device memory at outputPtr.
			/// The device memory must have been previously allocated and be large enough to hold all the results.
			/// Launches are done with the stream set using curandSetStream(), or the null stream if no stream has been set.
			/// Results are 32-bit floating point values with log-normal distribution based on
			/// an associated normal distribution with mean mean and standard deviation stddev.
			/// Normally distributed results are generated from pseudorandom generators with a Box-Muller transform,
			/// and so require n to be even.Quasirandom generators use an inverse cumulative distribution function
			/// to preserve dimensionality. The normally distributed results are transformed into log-normal distribution.
			/// There may be slight numerical differences between results generated on the GPU with generators created with
			/// curandCreateGenerator() and results calculated on the CPU with generators created with curandCreateGeneratorHost().
			/// These differences arise because of differences in results for transcendental functions.
			/// In addition, future versions of CURAND may use newer versions of the CUDA math library,
			/// so different versions of CURAND may give slightly different numerical values.
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="n">Number of floats to generate</param>
			/// <param name="mean">Mean of associated normal distribution</param>
			/// <param name="stddev">Standard deviation of associated normal distribution</param>
			/// <returns>
			/// CURAND_STATUS_NOT_INITIALIZED if the generator was never created
			/// CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch
			/// CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
			/// CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension, or is not a multiple of two for pseudorandom generators
			/// CURAND_STATUS_SUCCESS if the results were generated successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLogNormal(IntPtr generator, IntPtr outputPtr, size_t n, float mean, float stddev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLogNormalDouble(IntPtr generator, IntPtr outputPtr, size_t n, double mean, double stddev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLongLong(IntPtr generator, IntPtr outputPtr, size_t num);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateNormal(IntPtr generator, IntPtr outputPtr, size_t n, float mean, float stddev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateNormalDouble(IntPtr generator, IntPtr outputPtr, size_t n, double mean, double stddev);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGeneratePoisson(IntPtr generator, IntPtr outputPtr, size_t n, double lambda);

			/// <summary>
			/// Setup starting states.
			/// </summary>
			/// <remarks>
			/// Generate the starting state of the generator. This function is automatically called by generation functions
			/// such as curandGenerate() and curandGenerateUniform().
			/// It can be called manually for performance testing reasons to separate timings for starting state generation
			/// and random number generation.
			/// </remarks>
			/// <param name="generator">Generator to update</param>
			/// <returns>
			/// CURAND_STATUS_NOT_INITIALIZED if the generator was never created
			/// CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch
			/// CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
			/// CURAND_STATUS_SUCCESS if the seeds were generated successfully
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateSeeds(IntPtr generator);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateUniform(IntPtr generator, IntPtr outputPtr, size_t num);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateUniformDouble(IntPtr generator, IntPtr outputPtr, size_t num);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			static extern curandStatus curandGetDirectionVectors32(curandDirectionVectors32_t* vectors, curandDirectionVectorSet_t set);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			static extern curandStatus curandGetDirectionVectors64(curandDirectionVectors64_t* vectors, curandDirectionVectorSet_t set);
			*/

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetProperty(libraryPropertyType type, ref int value);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetScrambleConstants32(unsigned int** constants);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetScrambleConstants64(unsigned long long** constants);
			*/

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetVersion(ref int version);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetGeneratorOffset(IntPtr generator, ulong offset);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetGeneratorOrdering(IntPtr generator, curandOrdering order);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetQuasiRandomGeneratorDimensions(IntPtr generator, uint num_dimensions);

			/*
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetStream(IntPtr generator, cudaStream_t stream);
			*/

		}
		// ----- C# Interface

		IntPtr generator = IntPtr.Zero;
		ulong seed = 0;

		public ulong Seed {
			get { return seed; }
			set {
				seed = value;
				SetPseudoRandomGeneratorSeed(generator, value);
			}
		}

		public cuRAND(curandRngType type = curandRngType.CURAND_RNG_PSEUDO_DEFAULT, bool host = true) {
			if (host) {
				generator = CreateGeneratorHost(type);
			} else {
				generator = CreateGenerator(type);
			}
		}

		~cuRAND() {
			Dispose();
		}

		public void Dispose() {
			if (generator != IntPtr.Zero) {
				DestroyGenerator(generator);
				generator = IntPtr.Zero;
			}
		}

		public int[] Generate(int num) {
			int[] result = new int[num];
			int size = sizeof(int) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerate(generator, memory, num));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public float[] GenerateLogNormal(int num, float mean, float stddev) {
			float[] result = new float[num];
			int size = sizeof(float) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateLogNormal(generator, memory, num, mean, stddev));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public double[] GenerateLogNormalDouble(int num, double mean, double stddev) {
			double[] result = new double[num];
			int size = sizeof(double) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateLogNormalDouble(generator, memory, num, mean, stddev));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public long[] GenerateLong(int num) {
			long[] result = new long[num];
			int size = sizeof(long) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateLongLong(generator, memory, num));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public float[] GenerateNormal(int num, float mean, float stddev) {
			float[] result = new float[num];
			int size = sizeof(float) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateNormal(generator, memory, num, mean, stddev));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public double[] GenerateNormalDouble(int num, double mean, double stddev) {
			double[] result = new double[num];
			int size = sizeof(double) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateNormalDouble(generator, memory, num, mean, stddev));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public int[] GeneratePoisson(int num, double lambda) {
			int[] result = new int[num];
			int size = sizeof(int) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGeneratePoisson(generator, memory, num, lambda));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public float[] GenerateUniform(int num) {
			float[] result = new float[num];
			int size = sizeof(float) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateUniform(generator, memory, num));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public double[] GenerateUniformDouble(int num) {
			double[] result = new double[num];
			int size = sizeof(double) * num;
			IntPtr memory = Marshal.AllocHGlobal(size);

			CheckStatus(API.curandGenerateUniformDouble(generator, memory, num));

			Marshal.Copy(memory, result, 0, num);
			Marshal.FreeHGlobal(memory);
			return result;
		}

		public int GetProperty(libraryPropertyType type) {
			int result = 0;
			CheckStatus(API.curandGetProperty(type, ref result));
			return result;
		}

		public void SetGeneratorOffset(ulong offset) {
			CheckStatus(API.curandSetGeneratorOffset(generator, offset));
		}

		public void SetGeneratorOrdering(curandOrdering order) {
			CheckStatus(API.curandSetGeneratorOrdering(generator, order));
		}

		public void SetQuasiRandomGeneratorDimensions(uint num_dimensions) {
			CheckStatus(API.curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions));
		}

		IntPtr CreateGenerator(curandRngType type) {
			IntPtr generator = IntPtr.Zero;
			CheckStatus(API.curandCreateGenerator(ref generator, type));
			return generator;
		}

		IntPtr CreateGeneratorHost(curandRngType type) {
			IntPtr generator = IntPtr.Zero;
			CheckStatus(API.curandCreateGeneratorHost(ref generator, type));
			return generator;
		}

		void DestroyGenerator(IntPtr generator) {
			CheckStatus(API.curandDestroyGenerator(generator));
		}

		void SetPseudoRandomGeneratorSeed(IntPtr generator, ulong seed) {
			CheckStatus(API.curandSetPseudoRandomGeneratorSeed(generator, seed));
		}

		void GenerateSeeds(IntPtr generator) {
			CheckStatus(API.curandGenerateSeeds(generator));
		}

		public static int GetVersion() {
			int version = 0;
			CheckStatus(API.curandGetVersion(ref version));
			return version;
		}

		static void CheckStatus(curandStatus status) {
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new Exception(status.ToString());
			}
		}
	}

	/// <summary>
	/// CURAND choice of direction vector set
	/// </summary>
	public enum curandDirectionVectorSet {
		///<summary>Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.</summary>
		CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
		///<summary>Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.</summary>
		CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
		///<summary>Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions.</summary>
		CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
		///<summary>Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled.</summary>
		CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
	}

	/// <summary>
	/// CURAND ordering of results in memory
	/// </summary>
	public enum curandOrdering {
		///<summary>Best ordering for pseudorandom results.</summary>
		CURAND_ORDERING_PSEUDO_BEST = 100,
		///<summary>Specific default 4096 thread sequence for pseudorandom results.</summary>
		CURAND_ORDERING_PSEUDO_DEFAULT = 101,
		///<summary>Specific seeding pattern for fast lower quality pseudorandom results.</summary>
		CURAND_ORDERING_PSEUDO_SEEDED = 102,
		///<summary>Specific n-dimensional ordering for quasirandom results.</summary>
		CURAND_ORDERING_QUASI_DEFAULT = 201
	}

	/// <summary>
	/// CURAND generator types
	/// </summary>
	public enum curandRngType {
		CURAND_RNG_TEST = 0,
		///<summary>Default pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_DEFAULT = 100,
		///<summary>XORWOW pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_XORWOW = 101,
		///<summary>MRG32k3a pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_MRG32K3A = 121,
		///<summary>Mersenne Twister MTGP32 pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_MTGP32 = 141,
		///<summary>Mersenne Twister MT19937 pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_MT19937 = 142,
		///<summary>PHILOX-4x32-10 pseudorandom generator.</summary>
		CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
		///<summary>Default quasirandom generator.</summary>
		CURAND_RNG_QUASI_DEFAULT = 200,
		///<summary>Sobol32 quasirandom generator.</summary>
		CURAND_RNG_QUASI_SOBOL32 = 201,
		///<summary>Scrambled Sobol32 quasirandom generator.</summary>
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
		///<summary>Sobol64 quasirandom generator.</summary>
		CURAND_RNG_QUASI_SOBOL64 = 203,
		///<summary>Scrambled Sobol64 quasirandom generator.</summary>
		CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
	}

	/// <summary>
	/// CURAND function call status types
	/// </summary>
	public enum curandStatus {
		///<summary>No errors.</summary>
		CURAND_STATUS_SUCCESS = 0,
		///<summary>Header file and linked library version do not match.</summary>
		CURAND_STATUS_VERSION_MISMATCH = 100,
		///<summary>Generator not initialized.</summary>
		CURAND_STATUS_NOT_INITIALIZED = 101,
		///<summary>Memory allocation failed.</summary>
		CURAND_STATUS_ALLOCATION_FAILED = 102,
		///<summary>Generator is wrong type.</summary>
		CURAND_STATUS_TYPE_ERROR = 103,
		///<summary>Argument out of range.</summary>
		CURAND_STATUS_OUT_OF_RANGE = 104,
		///<summary>Length requested is not a multple of dimension.</summary>
		CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
		///<summary>GPU does not have double precision required by MRG32k3a.</summary>
		CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
		///<summary>Kernel launch failure.</summary>
		CURAND_STATUS_LAUNCH_FAILURE = 201,
		///<summary>Preexisting failure on library entry.</summary>
		CURAND_STATUS_PREEXISTING_FAILURE = 202,
		///<summary>Initialization of CUDA failed.</summary>
		CURAND_STATUS_INITIALIZATION_FAILED = 203,
		///<summary>Architecture mismatch, GPU does not support requested feature.</summary>
		CURAND_STATUS_ARCH_MISMATCH = 204,
		///<summary>Internal library error.</summary>
		CURAND_STATUS_INTERNAL_ERROR = 999

	}
}
