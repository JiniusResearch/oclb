#define STRIDE 512
#define NUM_READS 32

__kernel void Kadd(__global DATATYPE *input, __global DATATYPE *output)
{
    DATATYPE val;
    IDXTYPE gid = get_global_id(0);

    val = input[gid ];
//#pragma unroll
    for (int i = 0; i < 100; i++)
        //val = val << i | val >> (32-i);
        val = rotate(val,i);
    val += gid;
    output[gid] = val;
}
