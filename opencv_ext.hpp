#pragma once

#include <stdarg.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <optional>
#include <variant>
#include <regex>
#include <type_traits>
#include <vector>

#include <opencv2/opencv.hpp>

#define PRINT2BUFFER \
	va_list va; \
	va_start(va, format); \
	vsprintf_s(detail::buffer, detail::bufferSize, format, va); \
	va_end(va)

// global detail
namespace cv::detail
{
inline constexpr int bufferSize = 1024;
inline char buffer[bufferSize];

inline void print2Buffer(const char* format, ...)
{
	PRINT2BUFFER;
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
	seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}

namespace std
{
// Point_
template<typename _Tp>
struct less<cv::Point_<_Tp>>
{
	typedef cv::Point2f argument_type;
	typedef std::size_t result_type;

	bool operator()(const argument_type& p1, const argument_type& p2) const
	{
		return std::tie(p1.x, p1.y) < std::tie(p2.x, p2.y);
	}
};

template<typename _Tp>
struct hash<cv::Point_<_Tp>>
{
	typedef cv::Point2f argument_type;
	typedef std::size_t result_type;

	result_type operator()(const argument_type& p) const noexcept
	{
		result_type seed = 0;
		cv::detail::hash_combine(seed, p.x);
		cv::detail::hash_combine(seed, p.y);
		return seed;
	}
};
}

namespace cv
{
#ifdef HAVE_OPENCV_CORE
namespace detail
{
// type_traits
template<class T>
struct dim_vec : public std::is_arithmetic<T>
{
};

template<class T>
struct dim_vec<cv::Vec<T, 2>> : std::integral_constant<int, 2>
{
};

template<class T>
struct dim_vec<cv::Vec<T, 3>> : std::integral_constant<int, 3>
{
};

template<class T>
struct dim_vec<cv::Vec<T, 4>> : std::integral_constant<int, 4>
{
};

template<class T>
inline constexpr int dim_vec_v = dim_vec<T>::value;

}

namespace detail
{
template<class T, InterpolationFlags interpolation, class precision = float, class = std::enable_if_t<std::is_arithmetic_v<precision>>>
class interpolate
{
public:
	inline T operator()(const Mat& img, Point_<precision>&& pt) const
	{
		static_assert(false);
	}
};

template<class T, class precision>
class interpolate<T, INTER_NEAREST, precision>
{
public:
	inline T operator()(const Mat& img, Point_<precision>&& pt) const
	{
		return img.at<T>(Point(cvRound(pt.x), cvRound(pt.y)));
	}
};

template<class T, class precision>
class interpolate<T, INTER_LINEAR, precision>
{
public:
	inline T operator()(const Mat& img, Point_<precision>&& pt) const
	{
		constexpr precision one = static_cast<precision>(1.0);
		const int x_ = static_cast<int>(pt.x);
		const int y_ = static_cast<int>(pt.y);
		const precision dx = pt.x - x_;
		const precision dy = pt.y - y_;
#ifdef _DEBUG
		const int x[2] =
		{
			std::min(std::max(x_ + 0, 0), img.cols - 1),
			std::min(std::max(x_ + 1, 0), img.cols - 1)
		};
		const int y[2] =
		{
			std::min(std::max(y_ + 0, 0), img.rows - 1),
			std::min(std::max(y_ + 1, 0), img.rows - 1)
		};
		T f00 = img.at<T>(y[0], x[0]);
		T f10 = img.at<T>(y[0], x[1]);
		T f01 = img.at<T>(y[1], x[0]);
		T f11 = img.at<T>(y[1], x[1]);
#else
		T f00 = img.at<T>(y_ + 0, x_ + 0);
		T f10 = img.at<T>(y_ + 0, x_ + 1);
		T f01 = img.at<T>(y_ + 1, x_ + 0);
		T f11 = img.at<T>(y_ + 1, x_ + 1);
#endif
		T result;
		if constexpr (detail::dim_vec_v<T> == 1)
			result = (one - dx) * (one - dy) * f00 + dx * (one - dy) * f10 + (one - dx) * dy * f01 + dx * dy * f11;
		else if constexpr (detail::dim_vec_v<T> == 3)
		{
			for (int i = 0; i < 3; ++i)
				result[i] =  (one - dx) * (one - dy) * f00[i] + dx * (one - dy) * f10[i] + (one - dx) * dy * f01[i] + dx * dy * f11[i];
		}
		else
			assert(false);
		return result;
	}

protected:
	//inline precision interpolate1(precision w00, pre)
};

// to be rewriten
/*template<class T, class precision>
class interpolate<T, INTER_CUBIC, precision>
{
public:
	inline static T impl(const Mat& img, Point_<precision>&& pt)
	{
		static Mat coef = (Mat_<precision>(4, 4) <<
			1, 0, 0, 0,
			0, 0, 1, 0,
			-3, 3, -2, -1,
			2, -2, 1, 1
		);
		static Mat coef_T = (Mat_<precision>(4, 4) <<
			1, 0, -3, 2,
			0, 0, 3, -2,
			0, 1, -2, 1,
			0, 0, -1, 1
		);

		int x_ = (int)pt.x;
		int y_ = (int)pt.y;
		int x[4] =
		{
			std::max(x_ - 1, 0),
			x_,
			std::min(x_ + 1, img.cols),
			std::min(x_ + 2, img.cols),
		};
		int y[4] =
		{
			std:max(y_ - 1, 0),
			y_,
			std::min(y_ + 1, img.rows),
			std::min(y_ + 2, img.rows)
		};
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				f(i, j) = img.at<T>(y[j], x[i]);
		Mat A = (Mat_<precision>(4, 4) <<
			f(1, 1), f(2, 1), fy(1, 1), fy(2, 1),
			f(1, 2), f(2, 2), fy(1, 2), fy(2, 2),
			fx(1, 1), fx(2, 1), fxy(1, 1), fxy(2, 1),
			fx(1, 2), fx(2, 2), fxy(1, 2), fxy(2, 2)
		);
		Mat a = coef * A * coef_T;
		const precision dx = pt.x - T(x_);
		const precision dy = pt.y - T(y_);
		const precision one = precision(1);
		Mat x_para = (Mat_<precision>(1, 4) << one, dx, dx * dx, dx * dx * dx);
		Mat y_para = (Mat_<precision>(4, 1) << one, dy, dy * dy, dy * dy * dy);
		Mat result = (x_para * a * y_para);
		return static_cast<T>(result.at<precision>(0, 0));
	}

protected:
	inline static precision& f(int i, int j)
	{
		static precision f[4][4];
		return f[i][j];
	}

	inline static precision fx(int i, int j)
	{
		return (f(i + 1, j) - f(i - 1, j)) / static_cast<precision>(2);
	}

	inline static precision fy(int i, int j)
	{
		return (f(i, j + 1) - f(i, j - 1)) / static_cast<precision>(2);
	}

	inline static precision fxy(int i, int j)
	{
		return (f(i + 1, j + 1) + f(i - 1, j - 1) - f(i + 1, j - 1) - f(i - 1, j + 1)) / static_cast<precision>(4);
	}
};*/
}

// at a subpixel point
template<class T, InterpolationFlags interpolation = INTER_CUBIC, class precision>
inline T at(const Mat& img, const Point_<precision>& pt)
{
	return detail::interpolate<T, interpolation, precision>{}(img, std::move(pt));
}

template<class T, InterpolationFlags interpolation = INTER_CUBIC, class precision>
inline T at(const Mat& img, Point_<precision>&& pt)
{
	return detail::interpolate<T, interpolation, precision>{}(img, std::move(pt));
}

template<class T, InterpolationFlags interpolation, class precision>
inline precision at(const Mat& img, Point_<precision>&& pt, int channel)
{
	constexpr precision one = static_cast<precision>(1.0);
	const int x_ = static_cast<int>(pt.x);
	const int y_ = static_cast<int>(pt.y);
	const precision dx = pt.x - x_;
	const precision dy = pt.y - y_;
#ifdef _DEBUG
	const int x[2] =
	{
		std::min(std::max(x_ + 0, 0), img.cols - 1),
		std::min(std::max(x_ + 1, 0), img.cols - 1)
	};
	const int y[2] =
	{
		std::min(std::max(y_ + 0, 0), img.rows - 1),
		std::min(std::max(y_ + 1, 0), img.rows - 1)
	};
	precision f00 = img.at<T>(y[0], x[0])[channel];
	precision f10 = img.at<T>(y[0], x[1])[channel];
	precision f01 = img.at<T>(y[1], x[0])[channel];
	precision f11 = img.at<T>(y[1], x[1])[channel];
#else
	precision f00 = img.at<T>(y_ + 0, x_ + 0)[channel];
	precision f10 = img.at<T>(y_ + 0, x_ + 1)[channel];
	precision f01 = img.at<T>(y_ + 1, x_ + 0)[channel];
	precision f11 = img.at<T>(y_ + 1, x_ + 1)[channel];
#endif
	precision one_dx = one - dx;
	return static_cast<precision>((f00 * one_dx + f10 * dx) * (one - dy) + (f01 * one_dx + f11 * dx) * dy);
}

// at a region
inline Mat at(const Mat& img, int y, int x, const Size& s)
{
	return img(Rect(Point(x - s.width / 2, y - s.height / 2), s));
}

inline Mat at(const Mat& img, const Point& p, const Size& s)
{
	return img(Rect(Point(p.x - s.width / 2, p.y - s.height / 2), s));
}

// use Point to match Mat::at function
inline const bool in(const Point& p, const Size& size)
{
	return p.x >= 0 && p.x < size.width && p.y >= 0 && p.y < size.height;
}

inline const bool in(const Point& p, const Mat& mat)
{
	return p.x >= 0 && p.x < mat.cols && p.y >= 0 && p.y < mat.rows;
}

// operations on arrays

inline double median(const Mat& m, int histSize)
{
	// calcuate histogram
	float range[] = { 0, static_cast<float>(histSize) };
	const float* ranges = { range };
	cv::Mat hist;
	calcHist(&m, 1, 0, cv::Mat(), hist, 1, &histSize, &ranges);

	// compute median
	double medianVal;
	int sum = 0;
	int half = m.total() / 2;
	for (int i = 0; i < histSize; i++)
	{
		if ((sum += cvRound(hist.at<float>(i))) >= half)
		{
			medianVal = i; 
			break;
		}
	}
	return medianVal / histSize;
}

// drawing functions
struct Color
{
	constexpr Color(int r, int g, int b) : r(r), g(g), b(b), a(255)
	{
	}

	operator const Scalar() const
	{
		return Scalar(b, g, r, a);
	}

	operator const Vec3b() const
	{
		return Vec3b(b, g, r);
	}

	constexpr int operator[](size_t index) const
	{
		switch (index)
		{
		case 0:
			return b;
		case 1:
			return g;
		case 2:
			return r;
		case 3:
			return a;
		}

	}

	static Color Random()
	{
		static RNG_MT19937 rng;
		return Color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	}

	int r, g, b, a;
};

inline constexpr Color WHITE(255, 255, 255);
inline constexpr Color BLACK(0, 0, 0);
inline constexpr Color RED(255, 0, 0);
inline constexpr Color GREEN(0, 255, 0);
inline constexpr Color BLUE(0, 0, 255);
inline constexpr Color CYAN(0, 255, 255);
inline constexpr Color PURPLE(255, 0, 255);
inline constexpr Color YELLOW(255, 255, 0);
inline constexpr Color GRAY(127, 127, 127);

template<size_t N>
struct Palette
{
	Palette() = default;

	Palette(const std::initializer_list<std::optional<Color>>& p) : colors(p)
	{
	}

	Palette(const std::initializer_list<Color>& p)
	{
		std::copy(p.begin(), p.end(), colors.begin());
	}

	// constructor from only first N colors
	// TODO

	// constructor from initializer_list
	// TODO

	const std::optional<Color> operator[](size_t i) const
	{
		return i >= N ? std::nullopt : colors[i];
	}

	std::vector<std::optional<Color>> colors = std::vector<std::optional<Color>>(N);
};

template<class T>
inline void point(const Mat& img, const Point_<T>& p, const Scalar& color, int thickness = 1)
{
	cv::Point _p(p);
	line(img, _p, _p, color, thickness);
}

template<class T>
inline void points(const Mat& img, const std::vector<Point_<T>>& p, const Scalar& color, int thichness = 1)
{
	std::for_each(p.begin(), p.end(), std::bind(point<T>, img, std::placeholders::_1, color, thichness));
}

template<class T>
inline void cross(const Mat& img, const Point_<T>& pt, int size, const Scalar& color,
	int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	int x = cvRound(pt.x), y = cvRound(pt.y);
	line(img, Point(x, y - size / 2), Point(x, y + size / 2),
		color, thickness, lineType, shift);
	line(img, Point(x - size / 2, y), Point(x + size / 2, y),
		color, thickness, lineType, shift);
}

template<class T>
inline void cross(const Mat& img, const std::vector<Point_<T>>& pts, int size, const Scalar& color,
	int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	for (const Point_<T>& pt : pts)
		cross(img, pt, size, color, thickness, lineType, shift);
}

template<class T>
inline void crossX(const Mat& img, const Point_<T>& pt, int size, const Scalar& color,
	int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	int x = cvRound(pt.x), y = cvRound(pt.y);
	line(img, Point(x - size / 2, y - size / 2), Point(x + size / 2, y + size / 2),
		color, thickness, lineType, shift);
	line(img, Point(x - size / 2, y + size / 2), Point(x + size / 2, y - size / 2),
		color, thickness, lineType, shift);
}

template<class T>
inline void crossX(const Mat& img, const std::vector<Point_<T>>& pts, int size, const Scalar& color,
	int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	for (const Point_<T>& pt : pts)
		crossX(img, pt, size, color, thickness, lineType, shift);
}


template<class T>
inline void box(const Mat& img, const Point_<T>& pt, Size size, int length,
	const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	int x = cvRound(pt.x - size.width / T(2)), y = cvRound(pt.y - size.height / T(2));
	rectangle(img, cv::Rect(x, y, size.width, size.height), color, thickness, lineType, shift);
}

template<class T>
inline void aimingBox(const Mat& img, const Point_<T>& pt, Size size, int length,
	const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0)
{
	int x = cvRound(pt.x), y = cvRound(pt.y);
	line(img, Point(x - size.width / 2, y - size.height / 2),
		Point(x - size.width / 2 + length, y - size.height / 2), color, thickness, lineType, shift);
	line(img, Point(x - size.width / 2, y - size.height / 2),
		Point(x - size.width / 2, y - size.height / 2 + length), color, thickness, lineType, shift);
	line(img, Point(x + size.width / 2, y - size.height / 2),
		Point(x + size.width / 2 - length, y - size.height / 2), color, thickness, lineType, shift);
	line(img, Point(x + size.width / 2, y - size.height / 2),
		Point(x + size.width / 2, y - size.height / 2 + length), color, thickness, lineType, shift);
	line(img, Point(x - size.width / 2, y + size.height / 2),
		Point(x - size.width / 2 + length, y + size.height / 2), color, thickness, lineType, shift);
	line(img, Point(x - size.width / 2, y + size.height / 2),
		Point(x - size.width / 2, y + size.height / 2 - length), color, thickness, lineType, shift);
	line(img, Point(x + size.width / 2, y + size.height / 2),
		Point(x + size.width / 2 - length, y + size.height / 2), color, thickness, lineType, shift);
	line(img, Point(x + size.width / 2, y + size.height / 2),
		Point(x + size.width / 2, y + size.height / 2 - length), color, thickness, lineType, shift);
}

// text
namespace params::draw::text
{
inline int xOffset = 10;
inline int yOffset = 30;
inline int lineHeight = 15;
inline double fontScale = 0.5;
inline Color defaultColor = GREEN;
}

inline void drawText(const Mat& img, Point org, const Scalar& color, const double scale, const char* format, ...)
{
	PRINT2BUFFER;
	putText(img, detail::buffer, org, FONT_HERSHEY_SIMPLEX, scale, color);
}

template<class... T>
inline void drawText(const Mat& img, Point org, const Scalar& color, const char* format, const T&... t)
{
	drawText(img, org, color, params::draw::text::fontScale, format, t...);
}

namespace detail::draw::text
{
inline int line;
}

template<class... T>
inline void drawText(const Mat& img, int line, const Scalar& color, const char* format, const T&... t)
{
	drawText(img, Point(params::draw::text::xOffset, params::draw::text::yOffset + line * params::draw::text::lineHeight),
		color, format, t...);
}

template<class... T>
inline void drawText(const Mat& img, int line, const char* format, const T&... t)
{
	drawText(img, line, params::draw::text::defaultColor, format, t...);
}

inline void resetLine()
{
	detail::draw::text::line = -1;
}

inline void nextLine()
{
	++detail::draw::text::line;
}

template<class... T>
inline void nextLine(const Mat& img, const Scalar& color, const char* format, const T&... t)
{
	drawText(img, ++detail::draw::text::line, color, format, t...);
}

template<class... T>
inline void nextLine(const Mat& img, const char* format, const T&... t)
{
	nextLine(img, params::draw::text::defaultColor, format, t...);
}

inline void nextLine(const Mat& img, const Scalar& color, const std::string& str)
{
	drawText(img, ++detail::draw::text::line, color, str.c_str());
}

inline void nextLine(const Mat& img, const std::string& str)
{
	nextLine(img, params::draw::text::defaultColor, str);
}

template<class... T>
inline void firstLine(const Mat& img, const Scalar& color, const T&... t)
{
	resetLine();
	nextLine(img, color, t...);
}

template<class... T>
inline void firstLine(const Mat& img, const Color& color, const T&... t)
{
	resetLine();
	nextLine(img, static_cast<const Scalar&>(color), t...);
}

template<class... T>
inline void firstLine(const Mat& img, const Vec3b& color, const T&... t)
{
	resetLine();
	nextLine(img, static_cast<const Scalar&>(color), t...);
}

template<class... T>
inline void firstLine(const Mat& img, const T&... t)
{
	firstLine(img, static_cast<const Scalar&>(params::draw::text::defaultColor), t...);
}

// advanced drawing
template<class T>
Mat bar(std::vector<T> scores, int hIndex = -1, bool min = true, const Palette<4>& palette = {})
{
	const Vec3b background = palette[0].value_or(BLACK);
	const Vec3b foreground = palette[1].value_or(WHITE);
	const Vec3b highlight = palette[2].value_or(RED);
	const Vec3b highlight2 = palette[3].value_or(GREEN);
	int min_index = std::distance(scores.begin(),
		min ? std::min_element(scores.begin(), scores.end()) : std::max_element(scores.begin(), scores.end()));
	T max = *std::max_element(scores.begin(), scores.end());
	max += static_cast<T>(0.0001);
	std::for_each(scores.begin(), scores.end(), [&](T& f) { f /= max; });
	Mat bar(scores.size(), scores.size(), CV_8UC3);
	for (int y = 0; y < bar.rows; ++y)
	{
		T _y = static_cast<T>(1.0) - static_cast<T>(y) / bar.rows;
		for (int x = 0; x < bar.cols; ++x)
		{
			if (_y > scores[x])
				bar.at<Vec3b>(y, x) = background;
			else
			{
				if (x == min_index)
					bar.at<Vec3b>(y, x) = highlight;
				else if (x == hIndex)
					bar.at<Vec3b>(y, x) = highlight2;
				else
					bar.at<Vec3b>(y, x) = foreground;
			}
		}
	}
	return bar;
}
#endif

#ifdef HAVE_OPENCV_CALIB3D
inline double calibrate(const std::vector<Mat>& images, const Size& size, const double side,
	Mat& camera, Mat& dist)
{
	std::vector<Point3f> points;
	std::vector<Point2f> corners;
	for (int j = 0; j < size.height; ++j)
		for (int i = 0; i < size.width; ++i)
			points.emplace_back(Point3f(static_cast<float>(i * side), static_cast<float>(j * side), 0.0f));
	std::vector<std::vector<Point2f>> imagePoints;
	std::vector<std::vector<Point3f>> objectPoints;
	Mat gray;
	bool detected;
	for (size_t i = 0; i < images.size(); ++i)
	{
		if (images[i].channels() == 3)
			cvtColor(images[i], gray, COLOR_BGR2GRAY);
		else
			gray = images[i];
		detected = findChessboardCorners(gray, size, corners,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		if (detected)
		{
			cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
			imagePoints.push_back(corners);
			objectPoints.push_back(points);
		}
	}
	std::vector<Mat> rvecs, tvecs;
	return calibrateCamera(objectPoints, imagePoints, images[0].size(), camera, dist, rvecs, tvecs);
}
#endif

#ifdef HAVE_OPENCV_HIGHGUI
struct Grid
{
public:
	Grid() : width(0), height(0) {}
	Grid(int w, int h) : width(w), height(h) {}

	int width;
	int height;
};

inline Mat at(const Mat& img, const Grid& matrix, size_t index, const Grid& size = { 1, 1 })
{
	assert(index < size_t(matrix.width * matrix.height));
	size_t width = img.cols / matrix.width;
	size_t height = img.rows / matrix.height;
	return img(Rect(Point((index % matrix.width) * width, (index / matrix.width) * height),
		cv::Size(size.width * width, size.height * height)));
}

inline void imsubplot(Mat& img, const Grid& matrix, const Size& size, size_t index, const Mat& sub, bool resize = false)
{
	assert(index < size_t(matrix.width * matrix.height));
	if (sub.rows && sub.cols)
	{
		if (resize)
			cv::resize(sub, at(img, matrix, index), size);
		else
			sub.copyTo(at(img, matrix, index)(cv::Rect(cv::Point(0, 0), sub.size())));
	}
}

inline void imsubplot(Mat& img, const Grid& matrix, size_t index, const Mat& sub)
{
	assert(index < size_t(matrix.width * matrix.height));
	assert(sub.rows && sub.cols);
	const Size size = sub.size();
	sub.copyTo(at(img, matrix, index));
}

namespace params::imshow
{
struct for_each_subplot_tag
{
};

inline constexpr for_each_subplot_tag for_each_subplot {};

struct save_tag
{
};

inline constexpr save_tag save {};
}

namespace detail::impack
{
// find the maximum size of all the images
template<class... T>
inline void getMaxSize(Size& size)
{
}

template<class... T>
inline void getMaxSize(Size& size, const Mat& m, const T&... t);

template<class... T>
inline void getMaxSize(Size& size, InputOutputArray m, const T&... t)
{
	if (m.empty() == false)
	{
		const cv::Mat& mat = m.getMatRef();
		size.width = std::max(size.width, mat.cols);
		size.height = std::max(size.height, mat.rows);
	}
	getMaxSize(size, t...);
}

template<class... T>
inline void getMaxSize(Size& size, const Mat& m, const T&... t)
{
	size.width = std::max(size.width, m.cols);
	size.height = std::max(size.height, m.rows);
	getMaxSize(size, t...);
}

// find the type of the images
template<class... T>
inline int getType(const Mat& m, const T&... t);

template<class... T>
inline int getType(InputOutputArray m, const T&... t)
{
	if (m.empty() == false)
		return m.getMatRef().type();
	return getType(t...);
}

template<class... T>
inline int getType(const Mat& m, const T&... t)
{
	return m.type();
}

template<class... T>
inline void imsubplotFrom(Mat& img, const Grid& matrix, const Size& size, size_t index, const T&... t)
{
}

template<class... T>
inline void imsubplotFrom(Mat& img, const Grid& matrix, const Size& size, size_t index, const Mat& sub, const T&... t);

template<class... T>
inline void imsubplotFrom(Mat& img, const Grid& matrix, const Size& size, size_t index, InputOutputArray sub, const T&... t)
{
	if (sub.empty())
		imsubplotFrom(img, matrix, size, index + 1, t...);
	else
	{
		const cv::Mat& mat = sub.getMatRef();
		imsubplotFrom(img, matrix, size, index, mat, t...);
	}
}

template<class... T>
inline void imsubplotFrom(Mat& img, const Grid& matrix, const Size& size, size_t index, const Mat& sub, const T&... t)
{
	size_t x = index % matrix.width;
	size_t y = index / matrix.width;
	if (x < static_cast<size_t>(matrix.width) && y < static_cast<size_t>(matrix.height))
	{
		sub.copyTo(img(Rect(Point(x * size.width, y * size.height), sub.size())));
		imsubplotFrom(img, matrix, size, index + 1, t...);
	}
}
}

template<class... T>
inline Mat impack(const Grid& matrix, InputOutputArray sub1, const T&... t)
{
	Size size;
	detail::impack::getMaxSize(size, sub1, t...);
	Mat img = Mat::zeros(matrix.height * size.height, matrix.width * size.width, detail::impack::getType(sub1, t...));
	detail::impack::imsubplotFrom(img, matrix, size, 0, sub1, t...);
	return img;
}

template<class... T>
inline Mat impack(const Grid& matrix, const Size& size, const Mat& sub1, const T&... t)
{
	Mat img = Mat::zeros(matrix.height * size.height, matrix.width * size.width, detail::impack::getType(sub1, t...));
	detail::impack::imsubplotFrom(img, matrix, size, 0, sub1, t...);
	return img;
}

inline Mat impack(const Grid& matrix, const std::vector<Mat>& plots)
{
	assert(plots.size() > 0);
	Size size = plots[0].size();
	for (size_t i = 1; i < plots.size(); ++i)
	{
		size.width = std::max(size.width, plots[i].cols);
		size.height = std::max(size.height, plots[i].rows);
	}
	Mat img = Mat::zeros(size.height * matrix.height, size.width * matrix.width, plots[0].type());
	for (size_t i = 0; i < plots.size(); ++i)
		imsubplot(img, matrix, size, i, plots[i], false);
	return img;
}

inline Mat impack(const Grid& matrix, const Size& size, const std::vector<Mat>& plots)
{
	Mat img = Mat::zeros(size.height * matrix.height, size.width * matrix.width, plots[0].type());
	for (size_t i = 0; i < plots.size(); ++i)
		imsubplot(img, matrix, size, i, plots[i], false);
	return img;
}

namespace params::imshow
{
// some global variable
inline bool show = true;
inline double zoom = 1.0;
inline bool refresh = true;
inline bool wait = false;
inline bool defaultSave = false;
inline bool drawText = true;
inline std::string savePrefix;
inline std::string saveFormat = ".png";
}

namespace detail::imshow
{
template<class T>
using is_drawing_function = std::enable_if_t<std::is_invocable_v<T, Mat&>>;

inline int windowCount = 0;

struct Params
{
	Params()
	{
	}

	// first string is window name, all the rest are printed on the image
	// string from C style format string
	template<class... Tuple, class... T>
	Params(const std::tuple<Tuple...>& tuple, const T&... t) : Params(t...)
	{
		lines.emplace_back(std::apply(print, tuple));
	}

	// string from std::string
	template<class... T>
	Params(const std::string_view& s, const T&... t) : Params(t...)
	{
		lines.emplace_back(s);
	}

	// zoom
	template<class... T>
	Params(double z, const T&... t) : Params(t...)
	{
		zoom = z;
	}

	// mouse callback
	template<class... T>
	Params(MouseCallback cb, const T&... t) : Params(t...)
	{
		callback = cb;
	}

	// text color
	template<class... T>
	Params(const Color& color, const T&... t) : Params(t...)
	{
		textColor = color;
	}

	template<class Func, class... T, class = is_drawing_function<Func>>
	Params(Func fn, const T&... t) : Params(t...)
	{
		drawing.emplace_back(fn);
	}

	template<class... T>
	Params(const Grid& m, const T&... t) : Params(t...)
	{
		matrix = m;
	}

	template<class... T>
	Params(const Size& s, const T&... t) : Params(t...)
	{
		size = s;
	}

	template<class Func, class... T, class = is_drawing_function<Func>>
	Params(Func fn, params::imshow::for_each_subplot_tag, const T&... t) : Params(t...)
	{
		drawingSubplot.emplace_back(fn);
	}

	template<class... T>
	Params(params::imshow::save_tag, const T&... t) : Params(t...)
	{
		save = true;
	}

	// ignore all the images, won't be parsed as parameters
	template<class... T>
	Params(const Mat&, const T&... t) : Params(t...)
	{
	}

	template<class... T>
	Params(cv::InputOutputArray&, const T&... t) : Params(t...)
	{
	}

	std::optional<double> zoom = params::imshow::zoom;
	std::optional<MouseCallback> callback;
	std::optional<Color> textColor;
	std::vector<std::string> lines;
	std::vector<std::function<void(Mat&)>> drawing;
	Grid matrix;
	Size size;
	std::vector<std::function<void(Mat&)>> drawingSubplot;
	bool save = false;
protected:
	inline static std::string print(const char* format, ...)
	{
		PRINT2BUFFER;
		return buffer;
	}
};
}

template<class... T>
inline void imshow(const Mat& m, const T&... t)
{
	if (params::imshow::show == false)
		return;
	detail::imshow::Params p(t...);
	std::string winname;
	if (p.lines.size() && p.lines.back().length())
		winname = p.lines.back();
	else
	{
		// if no given window name
		detail::print2Buffer("Untitled %d", ++detail::imshow::windowCount);
		winname = detail::buffer;
	}
	if (p.callback)
	{
		namedWindow(winname);
		setMouseCallback(winname, *p.callback);
	}
	Mat out;
	if (p.zoom && *p.zoom != 1.0)
	{
		resize(m, out, Size(), *p.zoom, *p.zoom);
		if (out.channels() == 1)
			cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
	}
	else
	{
		if (m.channels() == 1)
			cv::cvtColor(m, out, cv::COLOR_GRAY2BGR);
		else
			out = m;
	}
	if (p.drawingSubplot.size())
	{
		for (int y = 0; y < p.matrix.height; ++y)
		{
			for (int x = 0; x < p.matrix.width; ++x)
			{
				Mat subplot = out(Rect(Point(x * p.size.width, y * p.size.height), p.size));
				for (auto& f : p.drawingSubplot)
					f(subplot);
			}
		}
	}
	for (auto& f : p.drawing)
		f(out);
	// text
	if (params::imshow::drawText)
	{
		const Color& c = p.textColor.value_or(params::draw::text::defaultColor);
		resetLine();
		if (p.lines.size() > 0)
			std::for_each(p.lines.rbegin() + 1, p.lines.rend(),
				std::bind(static_cast<void (*)(const Mat&, const Scalar&, const std::string&)>(nextLine),
					std::ref(out), c, std::placeholders::_1));
	}
	imshow(winname, out);
	if (params::imshow::refresh || params::imshow::wait)
		waitKey(!params::imshow::wait);
	if (params::imshow::defaultSave || p.save)
		imwrite("./" + params::imshow::savePrefix + "/" + winname + params::imshow::saveFormat, out);
}

template<class... T>
inline void imshow(const Grid& matrix, const T&... t)
{
	Mat pack = impack(matrix, t...);
	imshow(pack, matrix, t...);
}

template<class... T>
inline void imshow(const Grid& matrix, const std::vector<Mat>& plots, const T&... t)
{
	Mat pack = impack(matrix, plots);
	imshow(pack, matrix, t...);
}

struct loop_exit_tag {};
inline constexpr loop_exit_tag loop_exit;

template<typename T>
inline std::variant<T, loop_exit_tag> switch_key(T key)
{
	return key;
}

template<class T, class Func>
inline std::variant<T, loop_exit_tag> switch_key(T key, Func _default)
{
	_default(key);
	return key;
}

template<class T, class Func, class... Ts>
inline std::variant<T, loop_exit_tag> switch_key(T key, T candidate, Func func, const Ts&... ts);

template<class T, class Func, class... Ts, class = std::enable_if_t<std::is_integral_v<T>>>
inline std::variant<T, loop_exit_tag> switch_key(T key, const std::string& candidates, Func func, const Ts&... ts)
{
	if (candidates.find(key) != std::string::npos)
		return switch_key<T>(key, key, func, ts...);
	else
		return switch_key<T>(key, ts...);
}

template<class T, class Func, class... Ts, class = std::enable_if_t<std::is_integral_v<T>>>
inline std::variant<T, loop_exit_tag> switch_key(T key, const char* candidates, Func func, const Ts&... ts)
{
	return switch_key<T>(key, std::string(candidates), func, ts...);
}

template<class T, class Func, class... Ts>
inline std::variant<T, loop_exit_tag> switch_key(T key, const std::vector<T>& candidates, Func func, const Ts&... ts)
{
	if (std::find(candidates.begin(), candidates.end(), key) != candidates.end())
		return switch_key<T>(key, key, func, ts...);
	else
		return switch_key<T>(key, ts...);
}

template<class T, class Func, class... Ts>
inline std::variant<T, loop_exit_tag> switch_key(T key, T candidate, Func func, const Ts&... ts)
{
	if (key == candidate)
	{
		if constexpr (std::is_same_v<Func, loop_exit_tag>)
			return loop_exit;
		else
		{
			func();
			return key;
		}
	}
	else
		return switch_key<T>(key, ts...);
}

template<class Fn, class T = typename std::invoke_result_t<Fn>, class... Ts>
inline void loop(Fn func, const Ts&... ts)
{
	while (true)
	{
		std::variant<T, loop_exit_tag> result = switch_key<T>(func(), ts...);
		if (std::holds_alternative<loop_exit_tag>(result))
			return;
	}
}
#endif

#ifdef HAVE_OPENCV_IMGCODECS
namespace params::imwrite
{
inline static double fps = 30.0;
}

namespace detail::imwrite
{
inline static std::map<std::string, int> counters;
inline static std::map<std::string, cv::VideoWriter> videoWriters;
}

template<class... T>
inline void imwrite(const Mat& m, const char* format, T&&... t)
{
	static std::regex formatReg("[^%]*%[0-9]*d.*");
	static std::regex videoReg(".*(.mp4|.avi)");
	if (std::regex_match(format, formatReg) && sizeof...(t) == 0)
	{
		if (detail::imwrite::counters.count(format) == 0)
			detail::imwrite::counters[format] = 0;
		detail::print2Buffer(format, detail::imwrite::counters[format]++, std::forward<T>(t)...);
		imwrite(detail::buffer, m);
	}
	else if (std::regex_match(format, videoReg))
	{
		if (detail::imwrite::videoWriters.count(format) == 0)
		{
			cv::VideoWriter writer;
			if constexpr (sizeof...(t) == 0)
				writer.open(format, 0, params::imwrite::fps, m.size());
			else if constexpr (sizeof...(t) == 1)
				writer.open(format, 0, std::forward<T>(t)..., m.size());
			else if constexpr (sizeof...(t) == 2)
				writer.open(format, std::forward<T>(t)..., m.size());
			else if constexpr (sizeof...(t) == 3)
				writer.open(format, std::forward<T>(t)...);
			else
				assert(false);
			detail::imwrite::videoWriters[format] = std::move(writer);
		}
		detail::imwrite::videoWriters[format] << m;
	}
	else
	{
		if constexpr (sizeof...(t) == 0)
			imwrite(format, m);
		else
		{
			detail::print2Buffer(format, std::forward<T>(t)...);
			imwrite(detail::buffer, m);
		}
	}
}

template<class... T>
inline void imwrite(const Mat& m, const std::string& format, T&&... t)
{
	imwrite(m, format.c_str(), std::forward<T>(t)...);
}
#endif

#ifdef HAVE_OPENCV_IMGPROC
inline void cvtGray(const Mat& input, Mat& output)
{
	int c = input.channels();
	switch (c)
	{
	case 4:
		cvtColor(input, output, COLOR_BGRA2GRAY);
		break;
	case 3:
		cvtColor(input, output, COLOR_BGR2GRAY);
		break;
	case 1:
		output = input.clone();
	default:
		assert(false);
	}
}

inline void cvtBGR(const Mat& input, Mat& output)
{
	if (input.channels() == 1)
		cvtColor(input, output, COLOR_GRAY2BGR);
	else
		output = input.clone();
}
#endif
}

#undef PRINT2BUFFER