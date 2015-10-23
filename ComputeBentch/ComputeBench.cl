#define FORLOOP 1000

__kernel void Kadd(__global DATATYPE *output)
{
    DATATYPE val;
    IDXTYPE gid = get_global_id(0);

    val = gid;
    //val = (uint8)(gid, gid + 1, gid + 2, gid + 3,gid+4, gid + 5, gid + 6, gid + 7);

    //#pragma unroll
    for (uint i = 0; i < FORLOOP; i++) {
        //val = val << i | val >> (32-i);
        //        val.s0 = rotate(val.s0, i);
        //        val.s1 = rotate(val.s1, i);
        //        val.s2 = rotate(val.s2, i);
        //        val.s3 = rotate(val.s3, i);
        //val = rotate(val, i);


        
        val+=i;
        val = val^ i;
        val++;

        //val-=i;
        //val ^= i>>1;
    }
    output[gid] = val;
}
