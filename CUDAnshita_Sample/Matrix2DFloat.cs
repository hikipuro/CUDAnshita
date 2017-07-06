using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace CUDAnshita_Sample {
	public class Matrix2DFloat : IEnumerable<List<float>> {
#if NET20
		public delegate void Action();
		public delegate void Action<T1, T2>(T1 arg1, T2 arg2);
		public delegate void Action<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);
		public delegate void Action<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
		public delegate TResult Func<TResult>();
		public delegate TResult Func<T, TResult>(T arg);
		public delegate TResult Func<T1, T2, TResult>(T1 arg1, T2 arg2);
		public delegate TResult Func<T1, T2, T3, TResult>(T1 arg1, T2 arg2, T3 arg3);
		public delegate TResult Func<T1, T2, T3, T4, TResult>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
#endif
		protected float[] _Data;
		protected int _Width;
		protected int _Height;

		public float this[int i] {
			get { return _Data[i]; }
			set { _Data[i] = value; }
		}
		public float this[int x, int y] {
			get { return _Data[x + y * _Width]; }
			set { _Data[x + y * _Width] = value; }
		}

		public int Width {
			get { return _Width; }
		}
		public int Height {
			get { return _Height; }
		}
		public int Count {
			get { return _Data.Length; }
		}
		public float[] Data {
			get { return _Data; }
		}


		public static Matrix2DFloat operator -(Matrix2DFloat value) {
			var result = new Matrix2DFloat(value.Width, value.Height);
			result.ForEach((x, y) => {
				result[x, y] = -value[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator +(Matrix2DFloat value1, Matrix2DFloat value2) {
			if (value1 == null || value2 == null) {
				return null;
			}
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2DFloat(width, height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] + value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator +(Matrix2DFloat value1, float value2) {
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] + value2;
			});
			return result;
		}

		public static Matrix2DFloat operator +(float value1, Matrix2DFloat value2) {
			var result = new Matrix2DFloat(value2.Width, value2.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1 + value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator -(Matrix2DFloat value1, Matrix2DFloat value2) {
			if (value1 == null || value2 == null) {
				return null;
			}
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2DFloat(width, height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] - value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator -(Matrix2DFloat value1, float value2) {
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] - value2;
			});
			return result;
		}

		public static Matrix2DFloat operator -(float value1, Matrix2DFloat value2) {
			var result = new Matrix2DFloat(value2.Width, value2.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1 - value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator *(Matrix2DFloat value1, Matrix2DFloat value2) {
			var width = Math.Max(value1.Width, value2.Width);
			var height = Math.Max(value1.Height, value2.Height);
			var result = new Matrix2DFloat(width, height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] * value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator *(Matrix2DFloat value1, float value2) {
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] * value2;
			});
			return result;
		}

		public static Matrix2DFloat operator *(float value1, Matrix2DFloat value2) {
			var result = new Matrix2DFloat(value2.Width, value2.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1 * value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator /(Matrix2DFloat value1, float value2) {
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] / value2;
			});
			return result;
		}

		public static Matrix2DFloat operator /(float value1, Matrix2DFloat value2) {
			var result = new Matrix2DFloat(value2.Width, value2.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1 / value2[x, y];
			});
			return result;
		}

		public static Matrix2DFloat operator <(Matrix2DFloat value1, float value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] < value2 ? 1 : 0;
			});
			return result;
		}

		public static Matrix2DFloat operator >(Matrix2DFloat value1, float value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] > value2 ? 1 : 0;
			});
			return result;
		}

		public static Matrix2DFloat operator <=(Matrix2DFloat value1, float value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] <= value2 ? 1 : 0;
			});
			return result;
		}

		public static Matrix2DFloat operator >=(Matrix2DFloat value1, float value2) {
			if (value1 == null) {
				return null;
			}
			var result = new Matrix2DFloat(value1.Width, value1.Height);
			result.ForEach((x, y) => {
				result[x, y] = value1[x, y] >= value2 ? 1 : 0;
			});
			return result;
		}

		public Matrix2DFloat Equals(Matrix2DFloat value) {
			if (value == null) {
				return null;
			}
			var result = new Matrix2DFloat(Width, Height);
			result.ForEach((x, y) => {
				result[x, y] = this[x, y] == value[x, y] ? 1 : 0;
			});
			return result;
		}

		public Matrix2DFloat(int width, int height) {
			width = Math.Max(1, width);
			height = Math.Max(1, height);
			_Width = width;
			_Height = height;
			_Data = new float[width * height];
		}

		public Matrix2DFloat(int width) : this(width, 1) {
		}

		public Matrix2DFloat(List<float> list) {
			if (list == null) {
				throw new ArgumentNullException();
			}
			_Width = list.Count;
			_Height = 1;
			_Data = new float[_Width];
			for (int i = 0; i < _Width; i++) {
				_Data[i] = list[i];
			}
		}

		public Matrix2DFloat(float[] list) {
			if (list == null) {
				throw new ArgumentNullException();
			}
			_Width = list.Length;
			_Height = 1;
			_Data = new float[_Width];
			for (int i = 0; i < _Width; i++) {
				_Data[i] = list[i];
			}
		}

		public Matrix2DFloat Clone() {
			var result = new Matrix2DFloat(_Width, _Height);
			_Data.CopyTo(result._Data, 0);
			return result;
		}

		public static float Sum(Matrix2DFloat matrix) {
			float result = 0;
			matrix.ForEach((x, y) => {
				result = result + matrix[x, y];
			});
			return result;
		}

		public void Set(params float[] values) {
			var length = values.Length;
			for (var i = 0; i < length; i++) {
				if (i >= _Data.Length) {
					break;
				}
				_Data[i] = values[i];
			}
		}

		public List<int> Shape() {
			var result = new List<int>();
			result.Add(Width);
			result.Add(Height);
			return result;
		}

		public List<float> Flatten() {
			var result = new List<float>();
			ForEach((x, y) => {
				result.Add(this[x, y]);
			});
			return result;
		}

		public float Sum() {
			float result = 0;
			ForEach((x, y) => {
				result += this[x, y];
			});
			return result;
		}

		public float Max() {
			float result = 0;
			if (Count <= 0) {
				return result;
			}
			result = this[0, 0];
			ForEach((x, y) => {
				var value = this[x, y];
				if (value > result) {
					result = value;
				}
			});
			return result;
		}

		public int ArgMax() {
			var result = -1;
			if (Count <= 0) {
				return result;
			}
			var index = 0;
			var max = this[0, 0];
			result = 0;
			ForEach((x, y) => {
				var value = this[x, y];
				if (value > max) {
					max = value;
					result = index;
				}
				index++;
			});
			return result;
		}

		public Matrix2DFloat Dot(Matrix2DFloat value) {
			if (Width != value.Height) {
				throw new ArgumentException(string.Format(
					"ValueError: shapes ({0}, {1}) and ({2}, {3}) not aligned",
					Width, Height,
					value.Width, value.Height
				));
			}
			var result = new Matrix2DFloat(value.Width, Height);
			result.ForEach((x, y) => {
				float total = 0;
				int count = Width;
				for (int i = 0; i < count; i++) {
					var r = this[i, y] * value[x, i];
					total += r;
				}
				result[x, y] = total;
			});
			return result;
		}

		public Matrix2DFloat RandomChoice(int count) {
			if (count <= 0) {
				return null;
			}
			count = Math.Min(count, Height);
			var result = new Matrix2DFloat(Width, count);
			var index = new List<int>();
			for (var i = 0; i < Height; i++) {
				index.Add(i);
			}
			var random = new Random();
			for (var i = 0; i < Height; i++) {
				var n = random.Next(Height);
				var tmp = index[n];
				index[n] = index[i];
				index[i] = tmp;
			}
			for (var i = 0; i < count; i++) {
				var n = index[i];
				for (var x = 0; x < Width; x++) {
					result[x, i] = this[x, n];
				}
			}
			return result;
		}

		public void ForEach(Func<int, int, float, bool> action) {
			int width = _Width;
			int height = _Height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					if (action(x, y, this[x, y])) {
						return;
					}
				}
			}
		}

		public void ForEach(Action<int, int, float> action) {
			int width = _Width;
			int height = _Height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					action(x, y, this[x, y]);
				}
			}
		}

		public void ForEach(Action<int, int> action) {
			int width = _Width;
			int height = _Height;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					action(x, y);
				}
			}
		}

		public Matrix2DFloat Every(Func<float, float> predicate) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2DFloat(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(this[x, y]);
				}
			}
			return result;
		}

		public Matrix2DFloat Every(float left, Func<float, float, float> predicate) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2DFloat(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(left, this[x, y]);
				}
			}
			return result;
		}

		public Matrix2DFloat Every(Func<float, float, float> predicate, float right) {
			int width = _Width;
			int height = _Height;
			var result = new Matrix2DFloat(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					result[x, y] = predicate(this[x, y], right);
				}
			}
			return result;
		}

		public override string ToString() {
			var text = new StringBuilder();
			text.Append("[");

			int width = _Width;
			int height = _Height;
			for (var y = 0; y < height; y++) {
				if (height > 1) {
					text.AppendLine();
					text.Append("\t[");
				}
				for (var x = 0; x < width; x++) {
					text.Append(this[x, y].ToString());
					if (x + 1 != _Width) {
						text.Append(", ");
					}
				}
				if (height > 1) {
					text.Append("]");
				}
			}
			if (height > 1) {
				text.AppendLine();
			}
			text.Append("]");
			return text.ToString();
		}

		public List<float> Row(int y) {
			if (y >= _Height) {
				return null;
			}
			var data = new float[_Width];
			_Data.CopyTo(data, 0);
			return new List<float>(data);
		}

		public IEnumerator GetEnumerator() {
			int height = _Height;
			for (int y = 0; y < height; y++) {
				yield return Row(y);
			}
		}

		IEnumerator<List<float>> IEnumerable<List<float>>.GetEnumerator() {
			return (IEnumerator<List<float>>)GetEnumerator();
		}
	}
}

