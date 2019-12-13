#include <stdio.h>

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in $s at line $d\n",
               cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


template <typename T = int>
struct Matrix
{
    using Index = unsigned int;

    // 按行排列
    Matrix(const Index &width, const Index &height)
        : height(height), width(width), data(nullptr), dataCUDA(nullptr)
    {
        data = new T[width * height];
    }

    Matrix(const Index &n) : Matrix(n, n)
    {
    }

    ~Matrix()
    {
        if (data)
            delete[] data;
        
        if (dataCUDA)
            HANDLE_ERROR(cudaFree(dataCUDA));
    }

    // 读取或设置某个值
    T &operator()(const Index &row, const Index &col)
    {
        return data[col + row * width];
    }
    T getData(const Index &row, const Index &col)
    {
        return data[col + row * width];
    }
    auto getLength() const
    {
        return width * height;
    }
    auto getWidth() const
    {
        return width;
    }
    auto getHeight() const
    {
        return height;
    }
    auto printData(int maxWidth = 5, int maxHeight = 5, const char *printFormat = "%3.0f ")
    {
        int _width = width > maxWidth ? maxWidth : width,
            _height = height > maxHeight ? maxHeight : height;

        printf("width=%d, height=%d, type: %s\n", width, height, typeid(T).name());
        for (size_t row = 0; row < _height; ++row)
        {
            for (size_t col = 0; col < _width; ++col)
            {
                printf(printFormat, (*this)(row, col));
            }
            printf("\n");
        }
    }

    T *toCUDA()
    {
        if (dataCUDA == nullptr)
            HANDLE_ERROR(cudaMalloc((void **)&dataCUDA, getLength() * sizeof(T)));

        HANDLE_ERROR(cudaMemcpy(dataCUDA, data, getLength() * sizeof(T), cudaMemcpyHostToDevice));
        return dataCUDA;
    }
    T *toCPU(T *dataGPU = nullptr)
    {
        if (data == nullptr)
            data = new T[width * height];

        if (dataGPU)
            dataCUDA = dataGPU;

        HANDLE_ERROR(cudaMemcpy(data, dataCUDA, getLength() * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }

    T *data;
    T *dataCUDA;

private:
    Index width, height;
};