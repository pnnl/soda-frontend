// Copyright 2021 Battelle Memorial Institute

#include "util/args.hh"
#include "model.hh"

int main(int argc, const char *argv[])
{
    std::cout << "SODA FrontEnd. ./soda_frontend --help\n\n";
    SODA_FrontEnd::Args args(argc, argv);
    SODA_FrontEnd::Model model; 
    std::cout << args.getMLIRTestGenFolderName() << std::endl;
    if(args.getMLIRTestGenFolderName().empty())
    {
        SODA_FrontEnd::Model model_t(args.getArchJson(), 
                               args.getWeightH5(),
                               args.getGenMLIROutFn());
        model = std::move(model_t);                               
    }
    else
    {
        SODA_FrontEnd::Model model_t(args.getArchJson(), 
                               args.getWeightH5(),
                               args.getGenMLIROutFn(),
                               args.getMLIRTestGenFolderName());
        model = std::move(model_t);
    }
    model.MLIRGenerator();
    if (args.print_layers)
        model.printLayers();
}
