using System;
using System.Runtime.InteropServices;

namespace CUDAnshita {
	using curandGenerator_t = IntPtr;
	using curandRngType_t = curandRngType;
	using curandDiscreteDistribution_t = IntPtr;
	using curandDirectionVectors32_t = curandDirectionVectors32;
	using curandDirectionVectors64_t = curandDirectionVectors64;
	using curandDirectionVectorSet_t = curandDirectionVectorSet;
	using cudaStream_t = IntPtr;
	using size_t = Int64;

	/// <summary>
	/// The cuRAND library provides facilities that focus on the simple and
	/// efficient generation of high-quality pseudorandom and quasirandom numbers.
	/// </summary>
	/// <remarks>
	/// <a href="http://docs.nvidia.com/cuda/curand/">http://docs.nvidia.com/cuda/curand/</a>
	/// </remarks>
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
			/// <ul>
			/// <li>CURAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated</li>
			/// <li>CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU</li>
			/// <li>CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the dynamically linked library version</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid</li>
			/// <li>CURAND_STATUS_SUCCESS if generator was created successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreateGenerator(
				ref curandGenerator_t generator,
				curandRngType_t rng_type
			);

			/// <summary>
			/// Create new host CPU random number generator.
			/// </summary>
			/// <remarks>
			/// Creates a new host CPU random number generator of type rng_type and returns it in *generator.
			/// </remarks>
			/// <param name="generator">Pointer to generator</param>
			/// <param name="rng_type">Type of generator to create</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_ALLOCATION_FAILED if memory could not be allocated</li>
			/// <li>CURAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU</li>
			/// <li>CURAND_STATUS_VERSION_MISMATCH if the header file version does not match the dynamically linked library version</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the value for rng_type is invalid</li>
			/// <li>CURAND_STATUS_SUCCESS if generator was created successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreateGeneratorHost(
				ref curandGenerator_t generator,
				curandRngType_t rng_type
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandCreatePoissonDistribution(
				double lambda,
				ref curandDiscreteDistribution_t discrete_distribution
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandDestroyDistribution(
				curandDiscreteDistribution_t discrete_distribution
			);

			/// <summary>
			/// Destroy an existing generator.
			/// </summary>
			/// <remarks>
			/// Destroy an existing generator and free all memory associated with its state.
			/// </remarks>
			/// <param name="generator">Generator to destroy</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_SUCCESS if generator was destroyed successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandDestroyGenerator(
				curandGenerator_t generator
			);

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
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the generator is a 64 bit quasirandom generator. (use curandGenerateLongLong() with 64 bit quasirandom generators)</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerate(
				curandGenerator_t generator,
				uint[] outputPtr,
				size_t num
			);

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
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension, or is not a multiple of two for pseudorandom generators</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLogNormal(
				curandGenerator_t generator,
				float[] outputPtr,
				size_t n,
				float mean,
				float stddev
			);

			/// <summary>
			/// Generate log-normally distributed doubles.
			/// </summary>
			/// <remarks>
			/// Use generator to generate n double results into the device memory at outputPtr.
			/// The device memory must have been previously allocated and be large enough to hold all the results.
			/// Launches are done with the stream set using curandSetStream(), or the null stream if no stream has been set.
			/// Results are 64-bit floating point values with log-normal distribution
			/// based on an associated normal distribution with mean mean and standard deviation stddev.
			/// Normally distributed results are generated from pseudorandom generators with a Box-Muller transform,
			/// and so require n to be even.Quasirandom generators use an inverse cumulative distribution function
			/// to preserve dimensionality. The normally distributed results are transformed into log-normal distribution.
			/// There may be slight numerical differences between results generated on the GPU with generators
			/// created with curandCreateGenerator() and results calculated on the CPU with generators
			/// created with curandCreateGeneratorHost().
			/// These differences arise because of differences in results for transcendental functions.
			/// In addition, future versions of CURAND may use newer versions of the CUDA math library,
			/// so different versions of CURAND may give slightly different numerical values.
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="n">Number of doubles to generate</param>
			/// <param name="mean">Mean of normal distribution</param>
			/// <param name="stddev">Standard deviation of normal distribution</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension, or is not a multiple of two for pseudorandom generators</li>
			/// <li>CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLogNormalDouble(
				curandGenerator_t generator,
				double[] outputPtr,
				size_t n,
				double mean,
				double stddev
			);

			/// <summary>
			/// Generate 64-bit quasirandom numbers.
			/// </summary>
			/// <remarks>
			/// Use generator to generate num 64-bit results into the device memory at outputPtr.
			/// The device memory must have been previously allocated and be large enough to hold all the results.
			/// Launches are done with the stream set using curandSetStream(), or the null stream if no stream has been set.
			/// Results are 64-bit values with every bit random.
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="num">Number of random 64-bit values to generate</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the generator is not a 64 bit quasirandom generator</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateLongLong(
				curandGenerator_t generator,
				ulong[] outputPtr,
				size_t num
			);

			/// <summary>
			/// Generate normally distributed floats.
			/// </summary>
			/// <remarks>
			/// Use generator to generate n float results into the device memory at outputPtr.
			/// The device memory must have been previously allocated and be large enough to hold all the results.
			/// Launches are done with the stream set using curandSetStream(), or the null stream if no stream has been set.
			/// Results are 32-bit floating point values with mean mean and standard deviation stddev.
			/// Normally distributed results are generated from pseudorandom generators with a Box-Muller transform,
			/// and so require n to be even.Quasirandom generators use an inverse cumulative distribution function to preserve dimensionality.
			/// There may be slight numerical differences between results generated on the GPU with generators
			/// created with curandCreateGenerator() and results calculated on the CPU with generators
			/// created with curandCreateGeneratorHost().
			/// These differences arise because of differences in results for transcendental functions.In addition,
			/// future versions of CURAND may use newer versions of the CUDA math library,
			/// so different versions of CURAND may give slightly different numerical values.
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="n">Number of floats to generate</param>
			/// <param name="mean">Mean of normal distribution</param>
			/// <param name="stddev">Standard deviation of normal distribution</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension, or is not a multiple of two for pseudorandom generators</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateNormal(
				curandGenerator_t generator,
				float[] outputPtr,
				size_t n,
				float mean,
				float stddev
			);

			/// <summary>
			/// Generate normally distributed doubles.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="n">Number of doubles to generate</param>
			/// <param name="mean">Mean of normal distribution</param>
			/// <param name="stddev">Standard deviation of normal distribution</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension, or is not a multiple of two for pseudorandom generators</li>
			/// <li>CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateNormalDouble(
				curandGenerator_t generator,
				double[] outputPtr,
				size_t n,
				double mean,
				double stddev
			);

			/// <summary>
			/// Generate Poisson-distributed unsigned ints.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="n">Number of unsigned ints to generate</param>
			/// <param name="lambda">lambda for the Poisson distribution</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension</li>
			/// <li>CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU or sm does not support double precision</li>
			/// <li>CURAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGeneratePoisson(
				curandGenerator_t generator,
				uint[] outputPtr,
				size_t n,
				double lambda
			);

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
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_SUCCESS if the seeds were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateSeeds(
				curandGenerator_t generator
			);

			/// <summary>
			/// Generate uniformly distributed floats.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="num">Number of floats to generate</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateUniform(
				curandGenerator_t generator,
				float[] outputPtr,
				size_t num
			);

			/// <summary>
			/// Generate uniformly distributed doubles.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to use</param>
			/// <param name="outputPtr">Pointer to device memory to store CUDA-generated results, or Pointer to host memory to store CPU-generated results</param>
			/// <param name="num">Number of doubles to generate</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_PREEXISTING_FAILURE if there was an existing error from a previous kernel launch</li>
			/// <li>CURAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason</li>
			/// <li>CURAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is not a multiple of the quasirandom dimension</li>
			/// <li>CURAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision</li>
			/// <li>CURAND_STATUS_SUCCESS if the results were generated successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGenerateUniformDouble(
				curandGenerator_t generator,
				double[] outputPtr,
				size_t num
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetDirectionVectors32(
				ref IntPtr vectors,
				curandDirectionVectorSet_t set
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetDirectionVectors64(
				ref IntPtr vectors,
				curandDirectionVectorSet_t set
			);

			/// <summary>
			/// Return the value of the curand property.
			/// </summary>
			/// <remarks>
			/// Return in *value the number for the property described by type of the dynamically linked CURAND library.
			/// </remarks>
			/// <param name="type">CUDA library property</param>
			/// <param name="value">integer value for the requested property</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_SUCCESS if the property value was successfully returned</li>
			/// <li>CURAND_STATUS_OUT_OF_RANGE if the property type is not recognized</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetProperty(
				libraryPropertyType type,
				ref int value
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetScrambleConstants32(
				ref IntPtr constants
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetScrambleConstants64(
				ref IntPtr constants
			);

			/// <summary>
			/// Return the version number of the library.
			/// </summary>
			/// <remarks>
			/// Return in *version the version number of the dynamically linked CURAND library.
			/// The format is the same as CUDART_VERSION from the CUDA Runtime.
			/// The only supported configuration is CURAND version equal to CUDA Runtime version.
			/// </remarks>
			/// <param name="version">CURAND library version</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_SUCCESS if the version number was successfully returned</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandGetVersion(
				ref int version
			);

			/// <summary>
			/// Set the absolute offset of the pseudo or quasirandom number generator.
			/// </summary>
			/// <remarks>
			/// Set the absolute offset of the pseudo or quasirandom number generator.
			/// All values of offset are valid. The offset position is absolute,
			/// not relative to the current position in the sequence.
			/// </remarks>
			/// <param name="generator">Generator to modify</param>
			/// <param name="offset">Absolute offset position</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_SUCCESS if generator offset was set successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetGeneratorOffset(
				curandGenerator_t generator,
				ulong offset
			);

			/// <summary>
			/// Set the ordering of results of the pseudo or quasirandom number generator.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to modify</param>
			/// <param name="order">Ordering of results</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_OUT_OF_RANGE if the ordering is not valid</li>
			/// <li>CURAND_STATUS_SUCCESS if generator ordering was set successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetGeneratorOrdering(
				curandGenerator_t generator,
				curandOrdering order
			);

			/// <summary>
			/// Set the seed value of the pseudo-random number generator.
			/// </summary>
			/// <remarks>
			/// </remarks>
			/// <param name="generator">Generator to modify</param>
			/// <param name="seed">Seed value</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the generator is not a pseudorandom number generator</li>
			/// <li>CURAND_STATUS_SUCCESS if generator seed was set successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetPseudoRandomGeneratorSeed(
				curandGenerator_t generator,
				ulong seed
			);

			/// <summary>
			/// Set the number of dimensions.
			/// </summary>
			/// <remarks>
			/// Set the number of dimensions to be generated by the quasirandom number generator.
			/// Legal values for num_dimensions are 1 to 20000.
			/// </remarks>
			/// <param name="generator">Generator to modify</param>
			/// <param name="num_dimensions">Number of dimensions</param>
			/// <returns>
			/// <ul>
			/// <li>CURAND_STATUS_NOT_INITIALIZED if the generator was never created</li>
			/// <li>CURAND_STATUS_OUT_OF_RANGE if num_dimensions is not valid</li>
			/// <li>CURAND_STATUS_TYPE_ERROR if the generator is not a quasirandom number generator</li>
			/// <li>CURAND_STATUS_SUCCESS if generator ordering was set successfully</li>
			/// </ul>
			/// </returns>
			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetQuasiRandomGeneratorDimensions(
				curandGenerator_t generator,
				uint num_dimensions
			);

			[DllImport(DLL_PATH, CallingConvention = CALLING_CONVENTION)]
			public static extern curandStatus curandSetStream(
				curandGenerator_t generator,
				cudaStream_t stream
			);
		}

		// ----- C# Interface

		public static curandGenerator_t CreateGenerator(curandRngType_t rng_type) {
			curandGenerator_t generator = IntPtr.Zero;
			CheckStatus(API.curandCreateGenerator(ref generator, rng_type));
			return generator;
		}

		public static curandGenerator_t CreateGeneratorHost(curandRngType_t rng_type) {
			curandGenerator_t generator = IntPtr.Zero;
			CheckStatus(API.curandCreateGeneratorHost(ref generator, rng_type));
			return generator;
		}

		public static curandDiscreteDistribution_t CreatePoissonDistribution(double lambda) {
			curandDiscreteDistribution_t discrete_distribution = IntPtr.Zero;
			CheckStatus(API.curandCreatePoissonDistribution(lambda, ref discrete_distribution));
			return discrete_distribution;
		}

		public static void DestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
			CheckStatus(API.curandDestroyDistribution(discrete_distribution));
		}

		public static void DestroyGenerator(curandGenerator_t generator) {
			CheckStatus(API.curandDestroyGenerator(generator));
		}

		public static uint[] Generate(curandGenerator_t generator, size_t num) {
			if (num < 1) {
				return new uint[0];
			}
			uint[] outputPtr = new uint[num];
			CheckStatus(API.curandGenerate(generator, outputPtr, num));
			return outputPtr;
		}

		public static float[] GenerateLogNormal(curandGenerator_t generator, size_t n, float mean, float stddev) {
			if (n < 1) {
				return new float[0];
			}
			float[] outputPtr = new float[n];
			CheckStatus(API.curandGenerateLogNormal(generator, outputPtr, n, mean, stddev));
			return outputPtr;
		}

		public static double[] GenerateLogNormalDouble(curandGenerator_t generator, size_t n, double mean, double stddev) {
			if (n < 1) {
				return new double[0];
			}
			double[] outputPtr = new double[n];
			CheckStatus(API.curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev));
			return outputPtr;
		}

		public static ulong[] GenerateLongLong(curandGenerator_t generator, size_t num) {
			if (num < 1) {
				return new ulong[0];
			}
			ulong[] outputPtr = new ulong[num];
			CheckStatus(API.curandGenerateLongLong(generator, outputPtr, num));
			return outputPtr;
		}

		public static float[] GenerateNormal(curandGenerator_t generator, size_t n, float mean, float stddev) {
			if (n < 1) {
				return new float[0];
			}
			float[] outputPtr = new float[n];
			CheckStatus(API.curandGenerateNormal(generator, outputPtr, n, mean, stddev));
			return outputPtr;
		}

		public static double[] GenerateNormalDouble(curandGenerator_t generator, size_t n, double mean, double stddev) {
			if (n < 1) {
				return new double[0];
			}
			double[] outputPtr = new double[n];
			CheckStatus(API.curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev));
			return outputPtr;
		}

		public static uint[] GeneratePoisson(curandGenerator_t generator, size_t n, double lambda) {
			if (n < 1) {
				return new uint[0];
			}
			uint[] outputPtr = new uint[n];
			CheckStatus(API.curandGeneratePoisson(generator, outputPtr, n, lambda));
			return outputPtr;
		}

		public static void GenerateSeeds(curandGenerator_t generator) {
			CheckStatus(API.curandGenerateSeeds(generator));
		}

		public static float[] GenerateUniform(curandGenerator_t generator, size_t num) {
			if (num < 1) {
				return new float[0];
			}
			float[] outputPtr = new float[num];
			CheckStatus(API.curandGenerateUniform(generator, outputPtr, num));
			return outputPtr;
		}

		public static double[] GenerateUniformDouble(curandGenerator_t generator, size_t num) {
			if (num < 1) {
				return new double[0];
			}
			double[] outputPtr = new double[num];
			CheckStatus(API.curandGenerateUniformDouble(generator, outputPtr, num));
			return outputPtr;
		}

		public static int[] GetDirectionVectors32(curandDirectionVectorSet set) {
			IntPtr vectors = IntPtr.Zero;
			CheckStatus(API.curandGetDirectionVectors32(ref vectors, set));

			int[] result = new int[20000];
			Marshal.Copy(vectors, result, 0, result.Length);
			return result;
		}

		public static long[] GetDirectionVectors64(curandDirectionVectorSet set) {
			IntPtr vectors = IntPtr.Zero;
			CheckStatus(API.curandGetDirectionVectors64(ref vectors, set));

			long[] result = new long[20000];
			Marshal.Copy(vectors, result, 0, result.Length);
			return result;
		}

		public static int GetProperty(libraryPropertyType type) {
			int value = 0;
			CheckStatus(API.curandGetProperty(type, ref value));
			return value;
		}

		public static int[] GetScrambleConstants32() {
			IntPtr constants = IntPtr.Zero;
			CheckStatus(API.curandGetScrambleConstants32(ref constants));

			int[] result = new int[20000];
			Marshal.Copy(constants, result, 0, result.Length);
			return result;
		}

		public static long[] GetScrambleConstants64() {
			IntPtr constants = IntPtr.Zero;
			CheckStatus(API.curandGetScrambleConstants64(ref constants));

			long[] result = new long[20000];
			Marshal.Copy(constants, result, 0, result.Length);
			return result;
		}

		public static int GetVersion() {
			int version = 0;
			CheckStatus(API.curandGetVersion(ref version));
			return version;
		}

		public static void SetGeneratorOffset(curandGenerator_t generator, ulong offset) {
			CheckStatus(API.curandSetGeneratorOffset(generator, offset));
		}

		public static void SetGeneratorOrdering(curandGenerator_t generator, curandOrdering order) {
			CheckStatus(API.curandSetGeneratorOrdering(generator, order));
		}

		public static void SetPseudoRandomGeneratorSeed(curandGenerator_t generator, ulong seed) {
			CheckStatus(API.curandSetPseudoRandomGeneratorSeed(generator, seed));
		}

		public static void SetQuasiRandomGeneratorDimensions(curandGenerator_t generator, uint num_dimensions) {
			CheckStatus(API.curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions));
		}

		public static void SetStream(curandGenerator_t generator, cudaStream_t stream) {
			CheckStatus(API.curandSetStream(generator, stream));
		}

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

		public uint[] Generate(int num) {
			return Generate(generator, num);
		}

		public float[] GenerateLogNormal(int num, float mean, float stddev) {
			return GenerateLogNormal(generator, num, mean, stddev);
		}

		public double[] GenerateLogNormalDouble(int num, double mean, double stddev) {
			return GenerateLogNormalDouble(generator, num, mean, stddev);
		}

		public ulong[] GenerateLong(int num) {
			return GenerateLongLong(generator, num);
		}

		public float[] GenerateNormal(int num, float mean, float stddev) {
			return GenerateNormal(generator, num, mean, stddev);
		}

		public double[] GenerateNormalDouble(int num, double mean, double stddev) {
			return GenerateNormalDouble(generator, num, mean, stddev);
		}

		public uint[] GeneratePoisson(int num, double lambda) {
			return GeneratePoisson(generator, num, lambda);
		}

		public float[] GenerateUniform(int num) {
			return GenerateUniform(generator, num);
		}

		public double[] GenerateUniformDouble(int num) {
			return GenerateUniformDouble(generator, num);
		}

		public void SetGeneratorOffset(ulong offset) {
			SetGeneratorOffset(generator, offset);
		}

		public void SetGeneratorOrdering(curandOrdering order) {
			SetGeneratorOrdering(generator, order);
		}

		public void SetQuasiRandomGeneratorDimensions(uint num_dimensions) {
			SetQuasiRandomGeneratorDimensions(generator, num_dimensions);
		}

		public void SetStream(cudaStream_t stream) {
			SetStream(generator, stream);
		}

		static void CheckStatus(curandStatus status) {
			if (status != curandStatus.CURAND_STATUS_SUCCESS) {
				throw new CudaException(status.ToString());
			}
		}
	}

	[StructLayout(LayoutKind.Sequential, Size = 32 * 4)]
	public struct curandDirectionVectors32 {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
		public uint[] data;
	}

	[StructLayout(LayoutKind.Sequential, Size = 64 * 8)]
	public struct curandDirectionVectors64 {
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
		public ulong[] data;
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
