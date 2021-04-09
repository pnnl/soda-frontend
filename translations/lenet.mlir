// AutoGen - Do not modify
func @main() -> ()
{
    // Global register id starts at: 0
    // Layer type: Input
    // Layer tame: input_1
    // Input from Prev. layer: nan
    // Input size: 1 28 28 1 
    %0 = memref.alloc() : memref<1x28x28x1xf32>

    // Layer type: Conv2D
    // Layer name: conv2d
    // Input from layer: input_1
    // Input buffer: %0 : memref<1x28x28x1xf32>
    // Kernel dim.: 5 5 1 6 
    // Stride dim.: 1 1 
    // Dilation rates: 1 1 
    // Output size: 1 28 28 6 
    // Padding: [ 1 1 ] [ 1 1 ] 
    %1 = memref.alloc() : memref<5x5x1x6xf32>
    %2 = memref.alloc() : memref<1x28x28x6xf32>
    linalg.conv(%1, %0, %2)
    {
        dilations = [1, 1],
        padding = dense<[1, 1], [1, 1]> : tensor<2x2xi64>,
        strides = [1, 1]
    } : memref<5x5x1x6xf32>, memref<1x28x28x1xf32>, memref<1x28x28x6xf32>

    // Layer type: Activation
    // Layer name: activation
    // Input from layer: conv2d
    // Input buffer: %2 : memref<1x28x28x6xf32>
    // Output buffer: %2 : memref<1x28x28x6xf32>
    // Activation: relu
    scf.for %a = 0 to 1 step 1
    {
      scf.for %b = 0 to 28 step 1
      {
        scf.for %c = 0 to 28 step 1
        {
          scf.for %d = 0 to 6 step 1
          {
            %tmp = memref.load %2[%a, %b, %c, %d]  : memref<1x28x28x6xf32>
            %zero = constant 0.00000e+00 : f32
            %cond = cmpf "olt", %tmp, %zero : f32
            scf.if %cond
            {
              memref.store %zero, %2[%a, %b, %c, %d]  : memref<1x28x28x6xf32>
            }
          }
        }
      }
    }

    // Layer type: MaxPooling2D
    // Layer name: max_pooling2d
    // Input from layer: activation
    // Input buffer: %2 : memref<1x28x28x6xf32>
    // Kernel dim.: 1 2 2 1 
    // Stride dim.: 2 2 
    // Output size: 1 14 14 6 
    %3 = memref.alloc() : memref<1x2x2x1xf32>
    %4 = memref.alloc() : memref<1x14x14x6xf32>
    linalg.pooling_max(%2, %3, %4)
    {
        strides = [1, 2, 2, 1]
    } : memref<1x28x28x6xf32>, memref<1x2x2x1xf32>, memref<1x14x14x6xf32>

    // Layer type: Conv2D
    // Layer name: conv2d_1
    // Input from layer: max_pooling2d
    // Input buffer: %4 : memref<1x14x14x6xf32>
    // Kernel dim.: 5 5 6 16 
    // Stride dim.: 1 1 
    // Dilation rates: 1 1 
    // Output size: 1 10 10 16 
    // Padding: [ 0 0 ] [ 0 0 ] 
    %5 = memref.alloc() : memref<5x5x6x16xf32>
    %6 = memref.alloc() : memref<1x10x10x16xf32>
    linalg.conv(%5, %4, %6)
    {
        dilations = [1, 1],
        padding = dense<[0, 0], [0, 0]> : tensor<2x2xi64>,
        strides = [1, 1]
    } : memref<5x5x6x16xf32>, memref<1x14x14x6xf32>, memref<1x10x10x16xf32>

    // Layer type: Activation
    // Layer name: activation_1
    // Input from layer: conv2d_1
    // Input buffer: %6 : memref<1x10x10x16xf32>
    // Output buffer: %6 : memref<1x10x10x16xf32>
    // Activation: relu
    scf.for %a = 0 to 1 step 1
    {
      scf.for %b = 0 to 10 step 1
      {
        scf.for %c = 0 to 10 step 1
        {
          scf.for %d = 0 to 16 step 1
          {
            %tmp = memref.load %6[%a, %b, %c, %d]  : memref<1x10x10x16xf32>
            %zero = constant 0.00000e+00 : f32
            %cond = cmpf "olt", %tmp, %zero : f32
            scf.if %cond
            {
              memref.store %zero, %6[%a, %b, %c, %d]  : memref<1x10x10x16xf32>
            }
          }
        }
      }
    }

    // Layer type: MaxPooling2D
    // Layer name: max_pooling2d_1
    // Input from layer: activation_1
    // Input buffer: %6 : memref<1x10x10x16xf32>
    // Kernel dim.: 1 2 2 1 
    // Stride dim.: 2 2 
    // Output size: 1 5 5 16 
    %7 = memref.alloc() : memref<1x2x2x1xf32>
    %8 = memref.alloc() : memref<1x5x5x16xf32>
    linalg.pooling_max(%6, %7, %8)
    {
        strides = [1, 2, 2, 1]
    } : memref<1x10x10x16xf32>, memref<1x2x2x1xf32>, memref<1x5x5x16xf32>

    // Layer type: Flatten
    // Layer name: flatten
    // Input from layer: max_pooling2d_1
    // Input buffer: %8 : memref<1x5x5x16xf32>
    // Output size: 400
    %9 = memref.alloc() : memref<400xf32>
    scf.for %a = 0 to 1 step 1
    {
      scf.for %b = 0 to 5 step 1
      {
        scf.for %c = 0 to 5 step 1
        {
          scf.for %d = 0 to 16 step 1
          {
            %ld_val = memref.load %8[%a, %b, %c, %d]  : memref<1x5x5x16xf32>

            %index = addi %zero, %zero : i32

            %index_tmp  = muli %a, 400 : i32
            %index = addi %index_tmp, %index : i32

            %index_tmp  = muli %b, 80 : i32
            %index = addi %index_tmp, %index : i32

            %index_tmp  = muli %c, 16 : i32
            %index = addi %index_tmp, %index : i32

            %index = addi %d, %index : i32

            memref.store %ld_val, %9[%index] : memref<400xf32>
          }
        }
      }
    }

    // Layer type: Dense
    // Layer name: dense
    // Input from layer: flatten
    // Input buffer: %9 : memref<400xf32>
    // Kernel dim.: 400 500 
    // Output size: 500
    %10 = memref.alloc() : memref<400x500xf32>
    %11 = memref.alloc() : memref<500xf32>

    scf.for %a = 0 to 500 step 1
    {
      %out_val = addi %zero, %zero : i32
      scf.for %b = 0 to 400 step 1
      {
        %w_val = memref.load %10[%a]  : memref<400x500xf32>
        %in_val = memref.load %9[ : memref<400xf32>
        %out_tmp = mulf %in_val, %w_val : f32
        %out_val = addf %out_val, %out_tmp : f32
      }
      memref.store %out_val, %11[ : memref<500xf32>
    }

    // Layer type: Activation
    // Layer name: activation_2
    // Input from layer: dense
    // Input buffer: %11 : memref<500xf32>
    // Output buffer: %11 : memref<500xf32>
    // Activation: relu
    scf.for %a = 0 to 500 step 1
    {
      %tmp = memref.load %11[%a]  : memref<500xf32>
      %zero = constant 0.00000e+00 : f32
      %cond = cmpf "olt", %tmp, %zero : f32
      scf.if %cond
      {
        memref.store %zero, %11[%a]  : memref<500xf32>
      }
    }

    // Layer type: Dense
    // Layer name: dense_1
    // Input from layer: activation_2
    // Input buffer: %11 : memref<500xf32>
    // Kernel dim.: 500 10 
    // Output size: 10
    %12 = memref.alloc() : memref<500x10xf32>
    %13 = memref.alloc() : memref<10xf32>

    scf.for %a = 0 to 10 step 1
    {
      %out_val = addi %zero, %zero : i32
      scf.for %b = 0 to 500 step 1
      {
        %w_val = memref.load %12[%a]  : memref<500x10xf32>
        %in_val = memref.load %11[ : memref<500xf32>
        %out_tmp = mulf %in_val, %w_val : f32
        %out_val = addf %out_val, %out_tmp : f32
      }
      memref.store %out_val, %13[ : memref<10xf32>
    }

    // Layer type: Activation
    // Layre name: activation_3
    // Input from layer: dense_1
    // Input buffer: %13 : memref<10xf32>
    // Output buffer: %13 : memref<10xf32>
    // Activation: softmax
    // tmp buffer for exp eval 
    %14 = memref.alloc() : memref<10xf32>
    // buffer for result of softmax (exp norm) 
    %15 = memref.alloc() : memref<10xf32>
    %c0 = constant 0.0 :f32
    linalg.fill(%14, %c0) : memref<10xf32>, f32
    linalg.fill(%15, %c0) : memref<10xf32>, f32
    scf.for %a = 0 to 10 step 1
    {
      %tmp = memref.load %13[%a]  : memref<10xf32>
      %eval = math.exp %tmp : f32
      memref.store eval, %14[%a]  : memref<10xf32>
    }

    %16 = memref.alloc() : memref<0xf32>
    scf.for %a = 0 to 10 step 1 {
        scf.for %b = 0 to 0 step 1 {
          %rstmp = memref.load %14[%a]  : memref<10xf32>
          memref.store %rstmp, %16[%b]  : memref<0xf32>
        }
        %sum = scf.for %a = 0 to 0 step 1
          iter_args(%sum_itr = %sum_init) -> f32 {
          %stmp = memref.load %16[%a]  : memref<0xf32> 
          %sum_new = addf %sum_itr, %stmp : f32
          scf.yield %sum_new : f32
        }
        scf.for %b = 0 to 0 step 1 {
          %tnorm1 = memref.load %14[%a, %b]  : memref<10xf32>
          %tnorm2 = divf %tnorm1, %sum : f32
          memref.store %tnorm2, %15[%a, %b]  : memref<10xf32> 
        }
    }

    return;
}
