#include "helper_math.h"
#include "utils.h"


__device__ __forceinline__ float2 _ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 center,
    const float3 half_size
){

    const float3 t_min = (center-half_size-ray_o)*inv_d;
    const float3 t_max = (center+half_size-ray_o)*inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float t1 = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float t2 = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (t1 > t2) return make_float2(-1.0f); // no intersection
    return make_float2(t1, t2);
}


__global__ void ray_aabb_intersect_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> centers,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> half_sizes,
    const int max_hits,
    int* hit_cnt,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> hits_t,
    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> hits_voxel_idx
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (v>=centers.size(0) || r>=rays_o.size(0)) return;

    const float3 ray_o = make_float3(rays_o[r][0], rays_o[r][1], rays_o[r][2]);
    const float3 ray_d = make_float3(rays_d[r][0], rays_d[r][1], rays_d[r][2]);
    const float3 inv_d = 1.0f/ray_d;

    const float3 center = make_float3(centers[v][0], centers[v][1], centers[v][2]);
    const float3 half_size = make_float3(half_sizes[v][0], half_sizes[v][1], half_sizes[v][2]);
    const float2 t1t2 = _ray_aabb_intersect(ray_o, inv_d, center, half_size);

    if (t1t2.y > 0){ // if ray hits the voxel
        const int cnt = atomicAdd(&hit_cnt[r], 1);
        if (cnt < max_hits){
            hits_t[r][cnt][0] = fmaxf(t1t2.x, 0.0f);
            hits_t[r][cnt][1] = t1t2.y;
            hits_voxel_idx[r][cnt] = v;
        }
    }
}


std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
){

    const int N_rays = rays_o.size(0), N_voxels = centers.size(0);
    auto hits_t = torch::zeros({N_rays, max_hits, 2}, rays_o.options())-1;
    auto hits_voxel_idx = 
        torch::zeros({N_rays, max_hits}, 
                     torch::dtype(torch::kLong).device(rays_o.device()))-1;
    auto hit_cnt = 
        torch::zeros({N_rays}, 
                     torch::dtype(torch::kInt32).device(rays_o.device()));

    const dim3 threads(256, 1);
    const dim3 blocks((N_rays+threads.x-1)/threads.x,
                      (N_voxels+threads.y-1)/threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "ray_aabb_intersect_cu", 
    ([&] {
        ray_aabb_intersect_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            centers.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            half_sizes.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            max_hits,
            hit_cnt.data_ptr<int>(),
            hits_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hits_voxel_idx.packed_accessor64<long, 2, torch::RestrictPtrTraits>()
        );
    }));

    // sort intersections from near to far based on t1
    auto hits_order = std::get<1>(torch::sort(hits_t.index({"...", 0})));
    hits_voxel_idx = torch::gather(hits_voxel_idx, 1, hits_order);
    hits_t = torch::gather(hits_t, 1, hits_order.unsqueeze(-1).tile({1, 1, 2}));

    return {hit_cnt, hits_t, hits_voxel_idx};
}


__device__ __forceinline__ float2 _ray_sphere_intersect(
    const float3 ray_o,
    const float3 ray_d,
    const float3 center,
    const float radius
){
    const float3 co = ray_o-center;

    const float half_b = dot(ray_d, co);
    const float c = dot(co, co)-radius*radius;

    const float discriminant = half_b*half_b-c;

    if (discriminant < 0) return make_float2(-1.0f); // no intersection

    const float disc_sqrt = sqrtf(discriminant);
    return make_float2(-half_b-disc_sqrt, -half_b+disc_sqrt);
}


__global__ void ray_sphere_intersect_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> centers,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> radii,
    const int max_hits,
    int* hit_cnt,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> hits_t,
    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> hits_sphere_idx
){
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (s>=centers.size(0) || r>=rays_o.size(0)) return;

    const float3 ray_o = make_float3(rays_o[r][0], rays_o[r][1], rays_o[r][2]);
    const float3 ray_d = make_float3(rays_d[r][0], rays_d[r][1], rays_d[r][2]);
    const float3 center = make_float3(centers[s][0], centers[s][1], centers[s][2]);
    
    const float2 t1t2 = _ray_sphere_intersect(ray_o, ray_d, center, radii[s]);

    if (t1t2.y > 0){ // if ray hits the sphere
        const int cnt = atomicAdd(&hit_cnt[r], 1);
        if (cnt < max_hits){
            hits_t[r][cnt][0] = fmaxf(t1t2.x, 0.0f);
            hits_t[r][cnt][1] = t1t2.y;
            hits_sphere_idx[r][cnt] = s;
        }
    }
}

std::vector<torch::Tensor> ray_sphere_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
){

    const int N_rays = rays_o.size(0), N_spheres = centers.size(0);
    auto hits_t = torch::zeros({N_rays, max_hits, 2}, rays_o.options())-1;
    auto hits_sphere_idx = 
        torch::zeros({N_rays, max_hits}, 
                     torch::dtype(torch::kLong).device(rays_o.device()))-1;
    auto hit_cnt = 
        torch::zeros({N_rays}, 
                     torch::dtype(torch::kInt32).device(rays_o.device()));

    const dim3 threads(256, 1);
    const dim3 blocks((N_rays+threads.x-1)/threads.x,
                      (N_spheres+threads.y-1)/threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "ray_sphere_intersect_cu", 
    ([&] {
        ray_sphere_intersect_kernel<<<blocks, threads>>>(
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            centers.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            radii.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            max_hits,
            hit_cnt.data_ptr<int>(),
            hits_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hits_sphere_idx.packed_accessor64<long, 2, torch::RestrictPtrTraits>()
        );
    }));

    // sort intersections from near to far based on t1
    auto hits_order = std::get<1>(torch::sort(hits_t.index({"...", 0})));
    hits_sphere_idx = torch::gather(hits_sphere_idx, 1, hits_order);
    hits_t = torch::gather(hits_t, 1, hits_order.unsqueeze(-1).tile({1, 1, 2}));

    return {hit_cnt, hits_t, hits_sphere_idx};
}