#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <rapidjson/document.h>

// #include "triton/core/tritonserver.h"
#include "http_client.h"

namespace tc = triton::client;

int main() 
{
    // Create an HTTP client
    std::unique_ptr<tc::InferenceServerHttpClient> client;
    tc::InferenceServerHttpClient::Create(&client, "localhost:8000");
    bool live;
    client->IsServerLive(&live);
    if (!live) 
    {
        std::cerr << "Server is not live" << std::endl;
        return 1;
    }

    // Input data as std::vector<std::vector<double>>
    std::vector<std::vector<double>> input_data = {
        {1.5, 2.2, 3.3},
        {4.0, 57.2, 6.0},
        {7.0, 8.0, 9.0}
    };

    // Calculate total number of elements and flatten the 2D vector directly into uint64_t vector
    size_t total_elements = 0;
    for (const auto& row : input_data) 
    {
        total_elements += row.size();
    }
    std::vector<uint64_t> memory_buffer;
    memory_buffer.reserve(total_elements);
    for (const auto& row : input_data) 
    {
        for (double val : row) 
        {
            memory_buffer.push_back(static_cast<uint8_t>(val));
        }
    }

    // Prepare the input tensor
    std::vector<int64_t> shape = {static_cast<int64_t>(total_elements)};

    std::cout << "Shape: " << shape[0] << std::endl;

    tc::InferInput* input;
    auto err = tc::InferInput::Create(&input, "INPUT", shape, "UINT64");
    if (!err.IsOk()) 
    {
        std::cerr << "Error creating input: " << err << std::endl;
        return 1;
    }
    err = input->AppendRaw(reinterpret_cast<uint8_t*>(memory_buffer.data()),    
                                    memory_buffer.size() * sizeof(uint64_t));
    if (!err.IsOk()) 
    {
        std::cerr << "Error appending input data: " << err << std::endl;
        return 1;
    }

    // Prepare the output
    tc::InferRequestedOutput* output;
    err = tc::InferRequestedOutput::Create(&output, "OUTPUT");
    if (!err.IsOk()) 
    {
        std::cerr << "Error creating output: " << err << std::endl;
        return 1;
    }

    // Send inference request
    tc::InferOptions options("triton-minimal-backend");
    std::vector<tc::InferInput*> inputs = {input};
    std::vector<const tc::InferRequestedOutput*> outputs = {output};

    tc::InferResult* result;
    err = client->Infer(&result, options, inputs, outputs);
    if (!err.IsOk()) 
    {
        std::cerr << "Error sending inference request: " << err << std::endl;
        return 1;
    }

    // Get the result
    const uint8_t* output_data;
    size_t output_byte_size;
    err = result->RawData("OUTPUT", &output_data, &output_byte_size);
    if (!err.IsOk()) 
    {
        std::cerr << "Error getting inference result: " << err << std::endl;
        return 1;
    }

    // Assuming the original input data dimensions
    size_t num_rows = 3;
    size_t num_cols = 3;
    // Process the result (in this case, just print the first few elements)
    size_t output_size = output_byte_size / sizeof(double);
    std::vector<std::vector<double>> output_data_2d(num_rows, std::vector<double>(num_cols));

    const double* output_data_flat = reinterpret_cast<const double*>(output_data);
    for (size_t i = 0; i < num_rows; ++i) 
    {
        for (size_t j = 0; j < num_cols; ++j) 
        {
            output_data_2d[i][j] = output_data_flat[i * num_cols + j];
        }
    }

    // Print the reshaped output data
    std::cout << "Received output:" << std::endl;
    for (const auto& row : output_data_2d) 
    {
        for (const auto& elem : row) 
        {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    delete result;

    return 0;
}