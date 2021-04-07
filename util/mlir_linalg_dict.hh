#ifndef __MLIR_LINALG_DICT_HH__
#define __MLIR_LINALG_DICT_HH__

#include <string>
#include <unordered_map>
#include <vector>

namespace SODA_FrontEnd
{
namespace Linalg
{
#define UNDEFINED 0
#define ADDI 1
#define ALLOC 2
#define CMPLT 3
#define CONV2D 4
#define FOR 5
#define IF 6
#define LOAD 7
#define MAXPOOL 8
#define MULI 9
#define STORE 10

class MLIRDict
{
  public:

    std::unordered_map<unsigned,std::string> dict;

    MLIRDict()
    {
        dict.insert({UNDEFINED, "<UNDEFINED>"});
        dict.insert({ADDI, "addi"});
        dict.insert({ALLOC, "memref.alloc"});
        dict.insert({CMPLT, "cmpf \"olt\""});
        dict.insert({CONV2D, "linalg.conv"});
        dict.insert({FOR, "scf.for"});
        dict.insert({IF, "scf.if"});
        dict.insert({LOAD, "memref.load"});
        dict.insert({MAXPOOL, "linalg.pooling_max"});
        dict.insert({MULI, "muli"});
        dict.insert({STORE, "memref.store"});
    }

    std::string& operator[](unsigned opr)
    {
        if (auto opr_iter = dict.find(opr);
               opr_iter != dict.end())
        {
            return opr_iter->second;
        }
        else
        {
            return dict[UNDEFINED];
        }
    }
};
}
}

#endif
