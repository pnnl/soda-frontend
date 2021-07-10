// Copyright 2021 Battelle Memorial Institute

#ifndef __ARGS_HH__
#define __ARGS_HH__

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace SODA_FrontEnd
{
class Args
{
  public:
    std::string arch_json_fn;
    std::string weight_h5_fn;
    std::string gen_mlir_out;
    std::string mlir_test_gen_folder;
    bool print_layers; 

    auto &getArchJson() { return arch_json_fn; }
    auto &getWeightH5() { return weight_h5_fn; }
    auto &getGenMLIROutFn() { return gen_mlir_out; }
    auto &getMLIRTestGenFolderName() { return mlir_test_gen_folder; }

    Args(int argc, const char *argv[]): print_layers(0)
    {
        parseArgs(argc, argv);
    }

    void parseArgs(int argc, const char *argv[])
    {
        namespace po = boost::program_options;
	po::options_description desc("Options"); 
        desc.add_options() 
            ("help", "Print help messages")
            ("arch-json", po::value<std::string>(&arch_json_fn)->required(),
                "Architectural description in json format")
            ("weight-h5", po::value<std::string>(&weight_h5_fn)->required(),
                "Weight file in h5 format")
            ("mlir-gen", po::value<std::string>(&gen_mlir_out)->required(),
                "Generated MLIR output file")
            ("mlir-test-gen-folder", po::value<std::string>(&mlir_test_gen_folder),
                "Folder for MLIR test output files")
            ("print-layers", "Print layers and dimensions.");
        po::variables_map vm;
        try 
        { 
            po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

            if (vm.count("help")) 
            { 
                std::cout << "SODA front-end: \n" 
                          << desc << "\n";
                exit(0);
            }

            if (vm.count("print-layers"))
            {
                print_layers=true;
            }

            po::notify(vm);
        }
        catch(po::error& e) 
        {
            std::cerr << "ERROR: " << e.what() << "\n";
            std::cout << "SODA front-end: \n" 
                      << desc << "\n";
            exit(0);
        }
    }
};
}

#endif
