// Copyright 2021 Battelle Memorial Institute

#include "model.hh"

// boost library to parse json architecture file
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>

#include "util/mlir_linalg_gen.hh"
// #include "test/testLayers.hh"

namespace SODA_FrontEnd
{
void Model::Architecture::MLIRGenerator()
{
    for (auto i = 0; i < layers.size(); i++)
    {
        Layer &cur_layer = layers[i];
        cur_layer.setID(i);

        // std::cerr << "Cur. layer name: " << cur_layer.getName() << ", ID: " << i <<"\n";

        // Output input layers to the current layer
        if (auto map_iter = layer_inputs.find(cur_layer.getName());
                map_iter != layer_inputs.end())
        {

            auto &in_layer_names = map_iter->second;
            for (auto in_layer_name : in_layer_names)
            {
                Layer &in_layer = getLayer(in_layer_name);
                // std::cerr << "    Input layer: "
                //           << in_layer.getName() << "\n";
                
                // Store pointers to inbound layers in each layer
                if(cur_layer.getName() != "InputLayer") {
                    cur_layer.insertInLayer(&in_layer);
                }
            }
        }
    }

    /* TODO, temporarily disable the serial translation */

    // TODO, let's assume a serial connection
    Linalg::MLIRGen mlir_gen(mlir_gen_fn);
    mlir_gen.genInit(layers);
    // for (auto i = 0; i < layers.size(); i++)
    for (auto i = 0; i < 177; i++)
    {
        layers[i].setID(i);
        mlir_gen.genLayerBody(layers, i);
    }
    // Print Layer Names: 
    for (auto i = 0; i < layers.size(); i++) {
        mlir_gen.genPrintLayerId(i);
        mlir_gen.genPrintLayerName(layers[i]);
    }
    mlir_gen.genEnd();

    // SODA_FrontEnd::TestLayer testLayers(layers);
    Model::TestLayer testLayers(layers);
    std::cout << " -- Total Layers count: " << testLayers.numLayers() << std::endl;
    testLayers.binLayers();

    unsigned layer_id = 1;
    std::cout << mlir_test_gen_folder << std::endl;
    std::string test_file = mlir_test_gen_folder + "/" + layers[3].getName() + ".mlir";
    std::cout << test_file << std::endl;
    // testLayers.genTestLayer(testLayers.getLayer(), mlir_gen_fn);
    // testLayers.genTestLayer(layers, layer_id, mlir_gen_fn);

    for(int i=1; i < layers.size(); i++) 
    {
        // Model::Layer cur_layer = testLayers.getLayer(i);
        std::string layer_test_file_name = mlir_test_gen_folder + "/" + layers[i].getName() + ".mlir";
        testLayers.genTestLayer(layers, i, layer_test_file_name, mlir_gen.getVariableMap());
    }
    /* */
}

void Model::loadArch(std::string &arch_file)
{
    try
    {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(arch_file, pt);

        unsigned layer_counter = 0;
        // Iterate through the layers
        BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
                      pt.get_child("config.layers"))
        {
            // We need to construct the input layer first
            // Sometimes, input layer is not explicitly specified. 
            // When the input layer is explicitly specified, 
            //     we will change its name later.
            if (layer_counter == 0)
            {
                std::vector<std::string> input_shape;
                std::vector<unsigned> output_dims;
                for (boost::property_tree::ptree::value_type &cell : 
                    v.second.get_child("config.batch_input_shape"))
                {
                    auto val = cell.second.get_value<std::string>();
                    if (val == "null")
                    {
                        input_shape.push_back("1");
                    }
                    else
                    {
                        input_shape.push_back(val);
                    }
                }

                // input_shape.erase(input_shape.begin());
                for (auto dim : input_shape) { output_dims.push_back(stoll(dim)); }

                std::string name = "input";
                Layer::Layer_Type layer_type = Layer::Layer_Type::Input;
                arch.addLayer(name, layer_type);

                // TODO, set data type
                std::string d_type = v.second.get<std::string>("config.dtype");
                arch.getLayer(name).setDataType(d_type);
                arch.getLayer(name).setOutputDim(output_dims);

                // layer_counter++;
            }

            std::string class_name = v.second.get<std::string>("class_name");
            std::string name = v.second.get<std::string>("config.name");

            Layer::Layer_Type layer_type = Layer::Layer_Type::MAX;
            if (class_name == "InputLayer")
            {
                layer_type = Layer::Layer_Type::Input;
            }
            else if (class_name == "Conv2D")
            { 
                layer_type = Layer::Layer_Type::Conv2D; 
            }
            else if (class_name == "Activation")
            {
                layer_type = Layer::Layer_Type::Activation;
            }
            else if (class_name == "BatchNormalization") 
            {
                layer_type = Layer::Layer_Type::BatchNormalization;
            }
            else if (class_name == "Dropout") 
            {
                layer_type = Layer::Layer_Type::Dropout; 
            }
            else if (class_name == "MaxPooling2D")
            {
                layer_type = Layer::Layer_Type::MaxPooling2D;
            }
            else if (class_name == "AveragePooling2D")
            {
                layer_type = Layer::Layer_Type::AveragePooling2D;
            }
            else if (class_name == "Flatten") 
            { 
                layer_type = Layer::Layer_Type::Flatten;
            }
            else if (class_name == "Dense") 
            {
                layer_type = Layer::Layer_Type::Dense;
            }
            else if (class_name == "ZeroPadding2D") 
            {
                // break;
                layer_type = Layer::Layer_Type::ZeroPadding2D;
            }
            else if (class_name == "BatchNormalization") 
            {
                layer_type = Layer::Layer_Type::BatchNormalization;
            }
            else if (class_name == "Add") 
            {
                // break;
                layer_type = Layer::Layer_Type::Add;
            }
            else if (class_name == "GlobalAveragePooling2D") 
            {
                layer_type = Layer::Layer_Type::GlobalAveragePooling2D;
            }
            else 
            { 
                // std::cerr << "Error: Unsupported layer type.\n";
                std::cerr << class_name << " Not yet supported. \n";
                exit(0);
            }

            if (class_name != "InputLayer")
            {
                arch.addLayer(name, layer_type);
            }
            else if (class_name == "InputLayer")
            {
                // The input layer is explicitly specified, 
                // we need to change its name here.
                std::string default_name = "input";
                arch.getLayer(default_name).name = name;
            }

            std::string d_type = v.second.get<std::string>("config.dtype");
            arch.getLayer(name).setDataType(d_type);

            if (class_name == "Conv2D" || 
                class_name == "MaxPooling2D" || 
                class_name == "AveragePooling2D")
            {
                // get padding type
                std::string padding_type = v.second.get<std::string>("config.padding");
                if (padding_type == "same")
                {
                    arch.getLayer(name).padding_type = Layer::Padding_Type::same;
                }

                // get strides information
                std::vector<std::string> strides_str;
                std::vector<unsigned> strides;
                for (boost::property_tree::ptree::value_type &cell : 
                     v.second.get_child("config.strides"))
                {
                    strides_str.push_back(cell.second.get_value<std::string>());
                }
  
                for (auto stride : strides_str) { strides.push_back(stoll(stride)); }
                arch.getLayer(name).setStrides(strides);
            }

            if (class_name == "ZeroPadding2D")
            {
                auto &p_list = arch.getLayer(name).getPaddings();
                p_list.resize(0);
                
                // Count Children elements
                for (boost::property_tree::ptree::value_type &cell : 
                    v.second.get_child("config.padding"))
                {
                    boost::property_tree::ptree subt = cell.second;
                    for(boost::property_tree::ptree::value_type &cell2 : subt)
                    {
                        auto val = cell2.second.data();
                        p_list.push_back(std::stoul(val, nullptr, 0));
                        std::cout << std::string(val) << std::endl; 
                    }
                }
                arch.getLayer(name).setPaddings(p_list);
                auto &p_list2 = arch.getLayer(name).getPaddings();
                for(auto p : p_list2) 
                {
                    std::cout << p << std::endl;
                }
            }

            if (class_name == "Conv2D")
            {
                std::vector<std::string> dilations_str;
                std::vector<unsigned> dilations;
                for (boost::property_tree::ptree::value_type &cell : 
                     v.second.get_child("config.dilation_rate"))
                {
                    dilations_str.push_back(cell.second.get_value<std::string>());
                }
  
                for (auto d : dilations_str) { dilations.push_back(stoll(d)); }
                arch.getLayer(name).setDilations(dilations);
            }

            if (class_name == "MaxPooling2D" || 
                class_name == "AveragePooling2D")
            {
                // We need pool_size since 
                // Conv2D's kernel size can be extracted from h5 file
                std::vector<std::string> pool_size_str;
                auto &pool_size = arch.getLayer(name).w_dims;
                for (boost::property_tree::ptree::value_type &cell : 
                     v.second.get_child("config.pool_size"))
                {
                    pool_size_str.push_back(cell.second.get_value<std::string>());
                }

                pool_size.push_back(1);
                for (auto size : pool_size_str) { pool_size.push_back(stoll(size)); }
                pool_size.push_back(1); // depth is 1
            }

            if (class_name == "Activation")
            {
                std::string act = v.second.get<std::string>("config.activation");
                arch.getLayer(name).setActivation(act);
            }

            // TODO, I'm not sure if three-level are always true for 
            // all residual networks.
            std::vector<std::string> inbound_layers;
            for (boost::property_tree::ptree::value_type &l1 : 
                     v.second.get_child("inbound_nodes"))
            {
                for (boost::property_tree::ptree::value_type &l2 : 
                     l1.second)
                {
                    for (boost::property_tree::ptree::value_type &l3 : 
                         l2.second)
                    {
                        std::string inbound = l3.second.get_value<std::string>();
                        inbound_layers.push_back(inbound);

                        break; // TODO, we only consider the 1st ele
                    }
                }
            }

            arch.trackLayerConnection(name, inbound_layers);

            // get epsilon
            if (class_name == "BatchNormalization")
            {
                std::string eps = v.second.get<std::string>("config.epsilon");
                arch.getLayer(name).epsilon = std::stod(eps);
            }

            layer_counter++;
        }

        // Store pointers to inbound layers in each layer.
        // for( auto &layer : arch.getLayers())
        // {
        //     // std::cout << " -- " << layer.getName() << std::endl;
        //     if(layer.getName() != "InputLayer") 
        //     {
                

        //     }
        // }

    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
        exit(0);
    }
}

void Model::loadWeights(std::string &weight_file)
{
    // TODO, should be an exception handler for h5 file here

    // Example on parsing H5 format
    hid_t file;
    hid_t gid; // group id
    herr_t status;

    // char model_path[MAX_NAME];

    // Open h5 model
    file = H5Fopen(weight_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    gid = H5Gopen(file, "/", H5P_DEFAULT); // open root
    scanGroup(gid);

    status = H5Fclose(file);
}

void Model::scanGroup(hid_t gid)
{
    ssize_t len;
    hsize_t nobj;
    herr_t err;
    int otype;
    hid_t grpid, dsid;
    char group_name[MAX_NAME];
    char memb_name[MAX_NAME];
    char ds_name[MAX_NAME];

    // Get number of objects in the group
    len = H5Iget_name(gid, group_name, MAX_NAME);
    err = H5Gget_num_objs(gid, &nobj);

    // Iterate over every object in the group
    for (int i = 0; i < nobj; i++)
    {
        // Get object type
        len = H5Gget_objname_by_idx(gid, (hsize_t)i, memb_name, (size_t)MAX_NAME);
        otype = H5Gget_objtype_by_idx(gid, (size_t)i);

        switch (otype)
        {
            // If it's a group, recurse over it
        case H5G_GROUP:
            grpid = H5Gopen(gid, memb_name, H5P_DEFAULT);
            scanGroup(grpid);
            H5Gclose(grpid);
            break;
            // If it's a dataset, that means group has a bias and kernel dataset
        case H5G_DATASET:
            dsid = H5Dopen(gid, memb_name, H5P_DEFAULT);
            H5Iget_name(dsid, ds_name, MAX_NAME);
            // std::cout << ds_name << "\n";
            extrWeights(dsid);
            break;
        default:
            break;
        }
    }
}

void Model::extrWeights(hid_t id)
{
    hid_t datatype_id, space_id;
    herr_t status;
    hsize_t size;
    char ds_name[MAX_NAME];

    H5Iget_name(id, ds_name, MAX_NAME);
    space_id = H5Dget_space(id);
    datatype_id = H5Dget_type(id);

    // Get dataset dimensions to create buffer of same size
    const int ndims = H5Sget_simple_extent_ndims(space_id);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    // Calculating total 1D size
    unsigned data_size = 1;
    for (int i = 0; i < ndims; i++) { data_size *= dims[i]; }
    float *rdata = (float *)malloc(data_size * sizeof(float));
    status = H5Dread(id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    
    // Add information to the corres. layer
    std::stringstream full_name(ds_name);
    std::vector <std::string> tokens;
    std::string intermediate;

    while(getline(full_name, intermediate, '/'))
    {
        tokens.push_back(intermediate);
    }
    // The secondary last element indicates the layer name
    // TODO, I'm not sure if this is always true. Need to do more research
    if (tokens[tokens.size() - 1].find("kernel") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

	// for (int i = 0; i < ndims; i++) std::cout << dims[i] << " ";
        // std::cout << "\n";
        // std::cout << "kernel size: " << rdata_vec.size() << "\n\n";

        layer.setWeights(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("bias") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setBiases(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("beta") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        // for (int i = 0; i < ndims; i++) std::cout << dims[i] << " ";
        // std::cout << "\n";
        // std::cout << "beta size: " << rdata_vec.size() << "\n\n";

        layer.setBeta(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("gamma") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        // for (int i = 0; i < ndims; i++) std::cout << dims[i] << " ";
        // std::cout << "\n";
        // std::cout << "gamma size: " << rdata_vec.size() << "\n\n";

        layer.setGamma(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("moving_mean") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setMovingMean(dims_vec, rdata_vec);
    }
    else if (tokens[tokens.size() - 1].find("moving_variance") != std::string::npos)
    {
        Layer &layer = arch.getLayer(tokens[tokens.size() - 2]);
        std::vector<unsigned> dims_vec(dims, dims + ndims);
        std::vector<float> rdata_vec(rdata, rdata + data_size);

        layer.setMovingVariance(dims_vec, rdata_vec);
    }
    free(rdata);
}

// TestLayer
void Model::TestLayer::binLayers() 
{
    typedef Model::Layer::Layer_Type layer_enum_type; 
    typedef std::vector<Model::LayerMetaInfo> meta_lyr_vec_type; 
    assert(layers.size() != 0);
    // const auto& c_layers = layers; // Layers need to be constant as well. 
    for (auto&& cl : layers )
    {
        std::unordered_map<layer_enum_type, meta_lyr_vec_type>::iterator 
            map_itr = layer_meta_info.find(cl.getLayerType());
        
        layer_enum_type lt = cl.getLayerType(); 
        Model::LayerMetaInfo lyr_meta_info(cl.getID(), cl.getName(), lt);
        if(map_itr == layer_meta_info.end())
        {
            meta_lyr_vec_type meta_layer_vec {lyr_meta_info};
            layer_meta_info.insert({{lt, meta_layer_vec}});
        }
        else 
        {
            map_itr->second.emplace_back(lyr_meta_info);
        }
    }

    std::cout << " Layer Meta Info - Map size: " << layer_meta_info.size() << std::endl;
}

// Generate Single TestLayer
void Model::TestLayer::genTestLayer(std::vector<Model::Layer>& layers, unsigned layer_id, std::string mlir_layer_test_filename, std::map<std::string,int> variable_map) 
{
    // Test the first layer only 
    Linalg::MLIRGen mlir_gen_test(mlir_layer_test_filename);
    bool is_test = true; 
    mlir_gen_test.setTest(is_test);
    mlir_gen_test.setVariableMap(variable_map);
    if(!(layers[layer_id].getLayerType() == Layer::Layer_Type::Input))
    {
        mlir_gen_test.genInitLayerTest(layers[layer_id]);
        mlir_gen_test.genLayerBody(layers, layer_id);
        mlir_gen_test.genEnd();
    }
}

} // SODA_FRONTEND