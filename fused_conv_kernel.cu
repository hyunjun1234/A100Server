
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel for transposed convolution
template <typename scalar_t>
__global__ void conv_transpose_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width) {
    
    // Use 2D thread blocks for better spatial locality
    const int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    if (out_w >= output_width || out_h >= output_height || batch >= batch_size) {
        return;
    }
    
    // Cache weights in shared memory for the current output channel
    __shared__ scalar_t shared_weights[3 * 3 * 3]; // in_channels * kernel_size * kernel_size
    
    // Load weights into shared memory
    if (threadIdx.y < kernel_size && threadIdx.x < kernel_size && threadIdx.y * blockDim.x + threadIdx.x < in_channels * kernel_size * kernel_size) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    if (ic * kernel_size * kernel_size + kh * kernel_size + kw == threadIdx.y * blockDim.x + threadIdx.x) {
                        shared_weights[ic * kernel_size * kernel_size + kh * kernel_size + kw] = 
                            weight[ic * out_channels * kernel_size * kernel_size +
                                  out_c * kernel_size * kernel_size +
                                  (kernel_size - 1 - kh) * kernel_size + (kernel_size - 1 - kw)];
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    scalar_t result = 0.0f;
    
    #pragma unroll
    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int in_h = out_h - (kernel_size - 1) + kh;
                const int in_w = out_w - (kernel_size - 1) + kw;
                
                if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                    const int input_idx = batch * in_channels * input_height * input_width +
                                         ic * input_height * input_width +
                                         in_h * input_width + in_w;
                    
                    result += input[input_idx] * shared_weights[ic * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }
    
    const int output_idx = batch * out_channels * output_height * output_width +
                          out_c * output_height * output_width +
                          out_h * output_width + out_w;
    
    output[output_idx] = result;
}

// Optimized kernel for post-processing operations
template <typename scalar_t>
__global__ void post_processing_kernel(
    const scalar_t* __restrict__ conv_output,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ final_output,
    const int batch_size,
    const int out_channels,
    const int output_height,
    const int output_width) {
    
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size) return;
    
    const int pixels_per_channel = output_height * output_width;
    const scalar_t inv_pixels = 1.0f / pixels_per_channel;
    
    // Compute average pooling for each channel and add bias
    scalar_t channel_vals[16];  // out_channels = 16
    
    #pragma unroll
    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = 0.0f;
        
        const scalar_t* channel_data = conv_output + 
            batch * out_channels * pixels_per_channel + oc * pixels_per_channel;
        
        for (int i = 0; i < pixels_per_channel; ++i) {
            sum += channel_data[i];
        }
        
        // Average pooling and add bias
        channel_vals[oc] = sum * inv_pixels + bias[oc];
    }
    
    // Find max for numerical stability
    scalar_t max_val = channel_vals[0];
    #pragma unroll
    for (int oc = 1; oc < out_channels; ++oc) {
        max_val = max(max_val, channel_vals[oc]);
    }
    
    // Compute logsumexp with numerical stability
    scalar_t sum_exp = 0.0f;
    #pragma unroll
    for (int oc = 0; oc < out_channels; ++oc) {
        sum_exp += expf(channel_vals[oc] - max_val);
    }
    
    // Final result: log(sum(exp)) + max, then multiply by 10.0
    final_output[batch] = (logf(sum_exp) + max_val) * 10.0f;
}

torch::Tensor conv_transpose_fused_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    
    const auto out_channels = weight.size(1);
    const auto kernel_size = weight.size(2);
    
    const auto output_height = input_height + kernel_size - 1;
    const auto output_width = input_width + kernel_size - 1;
    
    // Allocate memory for convolution output
    auto conv_output = torch::zeros({batch_size, out_channels, output_height, output_width},
                                  input.options());
    
    // Allocate memory for final output
    auto final_output = torch::zeros({batch_size, 1},
                                   input.options());
    
    // Optimized grid and block configuration for convolution kernel
    const dim3 threads_conv(8, 8);
    const dim3 blocks_conv(
        (output_width + threads_conv.x - 1) / threads_conv.x,
        (output_height + threads_conv.y - 1) / threads_conv.y,
        batch_size * out_channels
    );
    
    // Optimized configuration for post-processing kernel
    const int threads_post = 128;
    const dim3 blocks_post((batch_size + threads_post - 1) / threads_post);
    
    // Launch kernels
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_fused_cuda", ([&] {
        conv_transpose_kernel<scalar_t><<<blocks_conv, threads_conv>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            conv_output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width);
        
        post_processing_kernel<scalar_t><<<blocks_post, threads_post>>>(
            conv_output.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            final_output.data_ptr<scalar_t>(),
            batch_size,
            out_channels,
            output_height,
            output_width);
    }));
    
    return final_output;
}
