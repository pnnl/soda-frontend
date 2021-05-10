#include "util/args.hh"
#include "model.hh"

int main(int argc, const char *argv[])
{
    std::cout << "SODA FrontEnd. ./soda_frontend --help\n\n";
    SODA_FrontEnd::Args args(argc, argv);
    SODA_FrontEnd::Model model(args.getArchJson(), 
                               args.getWeightH5(),
                               args.getGenMLIROutFn());
    if (args.print_layers)
        model.printLayers();
    model.MLIRGenerator();
}
