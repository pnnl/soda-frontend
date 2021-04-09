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
#define ADDF 1
#define ADDI 2
#define ALLOC 3
#define CMPLT 4
#define CONV2D 5
#define FOR 6
#define IF 7
#define LOAD 8
#define MAXPOOL 9
#define MULF 10
#define MULI 11
#define STORE 12
#define CONSTANT 13 
#define INDEX 14
#define FILLVAR 15
#define EXP 16 
#define YIELD 17
#define DIVF 18

class MLIRDict
{
  public:

    std::unordered_map<unsigned,std::string> dict;

    MLIRDict()
    {
        dict.insert({UNDEFINED, "<UNDEFINED>"});
        dict.insert({ADDF, "addf"});
        dict.insert({ADDI, "addi"});
        dict.insert({ALLOC, "memref.alloc"});
        dict.insert({CMPLT, "cmpf \"olt\""});
        dict.insert({CONV2D, "linalg.conv"});
        dict.insert({FOR, "scf.for"});
        dict.insert({IF, "scf.if"});
        dict.insert({LOAD, "memref.load"});
        dict.insert({MAXPOOL, "linalg.pooling_max"});
        dict.insert({MULF, "mulf"});
        dict.insert({MULI, "muli"});
        dict.insert({STORE, "memref.store"});
        dict.insert({CONSTANT, "constant"});
        dict.insert({INDEX, "index"});
        dict.insert({FILLVAR, "linalg.fill"});
        dict.insert({EXP, "math.exp"});
        dict.insert({YIELD, "scf.yield"});
        dict.insert({DIVF, "divf"});
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
