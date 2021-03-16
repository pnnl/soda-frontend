#include <stdlib.h>

#include "model.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Incorrect Usage\n";
        std::cout << "./main <arch_file> <weight_file>\n";
        return 1;
    }
    std::string arch_file(argv[1]);
    std::string weight_file(argv[2]);

    SODA_FrontEnd::Model model(arch_file, weight_file);
    
    model.printLayers();
}
