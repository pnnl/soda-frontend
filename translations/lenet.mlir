// AutoGen - Do not modify
func @main() -> ()
{
    // Global register id starts at: 0
    // Layer type: Input
    // Layer tame: input_1
    // Input from Prev. layer: nan
    // Input size: 1 28 28 1 
    %0 = memref.alloc() : memref<1x28x28x1xf32>

    // Layer Type: Conv2D
    // Layer Name: conv2d
    // Input from layer: input_1
    // Input buffer: %0 : memref<1x28x28x1xf32>
    // Kernel dim.: 5 5 1 6 
    // Stride dim.: 1 1 
    // Dilation rates: 1 1 
    // Output size: 1 28 28 6 
    // Padding: [ 2 2 ] [ 2 2 ] 
    %1 = memref.alloc() : memref<5x5x1x6xf32>
    %2 = memref.alloc() : memref<1x28x28x6xf32>
    linalg.conv(%1, %0, %2)
    {
        dilations = [1, 1],
        padding = dense<[2, 2], [2, 2]> : tensor<2x2xi64>,
        strides = [1, 1]
    } : memref<5x5x1x6xf32>, memref<1x28x28x1xf32>, memref<1x28x28x6xf32>

    // Layer Type: Activation
    // Layer Name: activation
    // Input from layer: conv2d
    // Input buffer: %2 : memref<1x28x28x6xf32>
    // Output buffer: %2 : memref<1x28x28x6xf32>
    // Activation: relu
    sci.for %a = 0 to 1 step 1
    {
      sci.for %b = 0 to 28 step 1
      {
        sci.for %c = 0 to 28 step 1
        {
          sci.for %d = 0 to 6 step 1
          {
            %tmp = load %2[%a, %b, %c, %d] : memref<1x28x28x6xf32>
            %zero = constant 0.00000e+00 : f32
            %cond = cmpf "olt", %tmp, %zero : f32
            sci.if %cond
            {
              store %zero,  %2[%a, %b, %c, %d] : memref<1x28x28x6xf32>
            }
          }
        }
      }
    }

    return;
}
