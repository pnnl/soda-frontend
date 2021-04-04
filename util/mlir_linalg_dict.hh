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
#define ALLOC 1
#define CMPLT 2
#define CONV2D 3
#define FOR 4
#define IF 5
#define LOAD 6
#define STORE 7

class MLIRDict
{
  public:

    std::unordered_map<unsigned,std::string> dict;

    MLIRDict()
    {
        dict.insert({UNDEFINED, "<UNDEFINED>"});
        dict.insert({ALLOC, "memref.alloc"});
        dict.insert({CMPLT, "cmpf \"olt\""});
        dict.insert({CONV2D, "linalg.conv"});
        dict.insert({FOR, "sci.for"});
        dict.insert({IF, "sci.if"});
        dict.insert({LOAD, "load"});
        dict.insert({STORE, "store"});
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
