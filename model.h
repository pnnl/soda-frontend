#ifndef __MODEL_H__
#define __MODEL_H__

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hdf5.h"

namespace SODA_FrontEnd
{
#define MAX_NAME 1024
class Model
{
  protected:
    // TODO, capture more information from json file.
    class Layer
    {
      public:
        // TODO, more layer type:
        // https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
	// https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
	// https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/

        // Batch-normalization: https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network
        enum class Layer_Type : int
        {
            Input, // the input layer
            Conv2D, // convolution layer
            Activation,
            BatchNormalization,
            Dropout,
            MaxPooling2D, // max polling layer
            AveragePooling2D,
            Flatten, // flatten layer
            Dense, // dense (fully-connected) layer
            MAX
        }layer_type = Layer_Type::MAX;

        Layer() {}
        Layer(std::string &_name, Layer_Type &_type) : name(_name), layer_type(_type) {}

        void setWeights(std::vector<unsigned> &_w_dims,
                        std::vector<float> &_weights)
        {
            w_dims = _w_dims;
            weights = _weights;
        }
        void setBiases(std::vector<unsigned> &_b_dims,
                       std::vector<float> &_bias)
        {
            b_dims = _b_dims;
            bias = _bias;
        }
        void setStrides(std::vector<unsigned> &_strides)
        {
            strides = _strides;
        }
        void setBeta(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            beta_dims = dims;
            beta = data;
        }
        void setGamma(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            gamma_dims = dims;
            gamma = data;
        }
        void setMovingMean(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_mean_dims = dims;
            moving_mean = data;
        }
        void setMovingVariance(std::vector<unsigned> &dims, std::vector<float> &data)
        {
            moving_variance_dims = dims;
            moving_variance = data;
        }
        void setOutputDim(std::vector<unsigned> &_dims)
        {
            output_dims = _dims;
        }

        std::string name; // Name of the layer

        // weights/biases for CONV2D/Dense
        std::vector<unsigned> w_dims; // dims of the weights
        std::vector<float> weights; // all the weight
        std::vector<unsigned> b_dims; // dims of the bias
        std::vector<float> bias; // all the bias

        // Padding type of the layer, used for CONV2D
        enum class Padding_Type : int
        {
            same,
            valid
        }padding_type = Padding_Type::valid;
        // strides, used for CONV2D/MaxPooling/AveragePooling
        std::vector<unsigned> strides;

        // TODO, need to extract more information
        // For batch-normalization
        std::vector<unsigned> beta_dims;
        std::vector<float> beta;
        std::vector<unsigned> gamma_dims;
        std::vector<float> gamma;
        std::vector<unsigned> moving_mean_dims;
        std::vector<float> moving_mean;
        std::vector<unsigned> moving_variance_dims;
        std::vector<float> moving_variance;

	std::vector<unsigned> output_dims; // dimension of output
    };

    // Model - Architecture
    class Architecture
    {
      protected:
        std::vector<Layer> layers;

      public:
        Architecture() {}

        void addLayer(std::string &_name, Layer::Layer_Type &_type)
        {
            layers.emplace_back(_name, _type);
        }

        Layer& getLayer(std::string &name)
        {
            for (auto &layer : layers)
            {
                if (layer.name == name) { return layer; }
            }
            std::cout << name << "\n";
            std::cerr << "Error: layer is not found.\n";
            exit(0);
        }

        void printLayers() // Only used for small network debuggings.
        {
            for (auto &layer : layers)
            {
                auto name = layer.name;
                auto type = layer.layer_type;

                std::cout << "Layer name: " << name << "; ";
                if (type == Layer::Layer_Type::Input) { std::cout << "Layer type: Input"; }
		else if (type == Layer::Layer_Type::Conv2D) { std::cout << "Layer type: Conv2D"; }
                else if (type == Layer::Layer_Type::Activation) { std::cout << "Layer type: Activation"; }
                else if (type == Layer::Layer_Type::BatchNormalization) { std::cout << "Layer type: BatchNormalization"; }
                else if (type == Layer::Layer_Type::Dropout) { std::cout << "Layer type: Dropout"; }
                else if (type == Layer::Layer_Type::MaxPooling2D) { std::cout << "Layer type: MaxPooling2D"; }
                else if (type == Layer::Layer_Type::AveragePooling2D) { std::cout << "Layer type: AveragePooling2D"; }
                else if (type == Layer::Layer_Type::Flatten) { std::cout << "Layer type: Flatten"; }
                else if (type == Layer::Layer_Type::Dense) { std::cout << "Layer type: Dense"; }
                else { std::cerr << "Error: unsupported layer type\n"; exit(0); }
                std::cout << "\n";
              
                if (type == Layer::Layer_Type::Input)
                {
                    auto &output_dims = layer.output_dims;
                    std::cout << "Input shape: ";
                    for (auto dim : output_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                auto &w_dims = layer.w_dims;
                auto &weights = layer.weights;
                auto &b_dims = layer.b_dims;
                auto &bias = layer.bias;
                if (weights.size())
                {
                    std::cout << "Weights dim (" << weights.size() << "): ";
                    for (auto dim : w_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                if (bias.size())
                {
                    std::cout << "Bias dim (" << bias.size() << "): ";
                    for (auto dim : b_dims) { std::cout << dim << " "; }
                    std::cout << "\n";
                }

                std::cout << "Total params: " << weights.size() + bias.size() << "\n";

                std::cout << "\n";
            }
        }
    };

    Architecture arch;

  public:
    Model(std::string &arch_file, std::string &weight_file)
    {
        loadArch(arch_file);
        loadWeights(weight_file);
    }

    void printLayers() { arch.printLayers(); }

  protected:
    void loadArch(std::string &arch_file);
    void loadWeights(std::string &weight_file);

  protected:
    void scanGroup(hid_t);
    void extrWeights(hid_t);
};
}

#endif
