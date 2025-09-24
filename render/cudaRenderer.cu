#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

# define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s at %s: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

#define TILE_H 32
#define TILE_W 32
#define CIRCLES_PER_CHUNK 128

#define SCAN_BLOCK_DIM 256

#include "exclusiveScan.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float4* imagePtr, int circleIndex) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * index;

    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // For all pixels in the bounding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(pixelCenterNorm, p, imgPtr, index);
            imgPtr++;
        }
    }
}

__device__ __forceinline__ int clamp(int val, int minVal, int maxVal)
{
    return (val < minVal) ? minVal : ((val > maxVal) ? maxVal : val);
}


__global__ void kernelTileCounts(
    int tilesX, int tilesY,
    int* __restrict__ tileCounts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cuConstRendererParams.numberOfCircles)
        return;

    float3 p = *(float3*)(&cuConstRendererParams.position[3 * idx]);
    float r = cuConstRendererParams.radius[idx];
    int imageW = cuConstRendererParams.imageWidth;
    int imageH = cuConstRendererParams.imageHeight;

    int minX = static_cast<int>(floorf(imageW * (p.x - r)));
    int maxX = static_cast<int>(ceilf(imageW * (p.x + r)));
    int minY = static_cast<int>(floorf(imageH * (p.y - r)));
    int maxY = static_cast<int>(ceilf(imageH * (p.y + r)));

    int txmin = clamp(minX / TILE_W, 0, tilesX - 1);
    int txmax = clamp(maxX / TILE_W, 0, tilesX - 1);
    int tymin = clamp(minY / TILE_H, 0, tilesY - 1);
    int tymax = clamp(maxY / TILE_H, 0, tilesY - 1);

    for (int ty = tymin; ty <= tymax; ty++) {
        int base = ty * tilesX;
        for (int tx = txmin; tx <= txmax; tx++) {
            atomicAdd(&tileCounts[base + tx], 1);
        }
    }
}


__global__ void kernelSortedTileList(
    int tilesX, int tilesY, int numCircles,
    const int* __restrict__ tileOffsets,
    int* __restrict__ tileCounts,
    int* __restrict__ tileList)
{
    const int b = blockIdx.x;
    if (b >= tilesX * tilesY) return;
    const int tx = threadIdx.x;

    const int imageW = cuConstRendererParams.imageWidth;
    const int imageH = cuConstRendererParams.imageHeight;
    const int bx = b % tilesX;
    const int by = b / tilesX;
    const float minx = bx * TILE_W;
    const float maxx = fminf((bx + 1) * TILE_W, (float)imageW);
    const float miny = by * TILE_H;
    const float maxy = fminf((by + 1) * TILE_H, (float)imageH);

    __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumScratch[2 * SCAN_BLOCK_DIM];
    __shared__ int tempCount;

    int base = 0;
    for (int cbase = 0; cbase < numCircles; cbase += SCAN_BLOCK_DIM) {
        const int c = cbase + tx;

        uint flag = 0u;
        if (c < numCircles) {
            float3 p = *(float3*)(&cuConstRendererParams.position[3 * c]);
            float r = cuConstRendererParams.radius[c];

            const float cx = imageW * p.x;
            const float cy = imageH * p.y;
            const float cr = imageW * r;

            float nx = fminf(fmaxf(cx, minx), maxx);
            float ny = fminf(fmaxf(cy, miny), maxy);
            float dx = cx - nx, dy = cy - ny;
            if (dx*dx + dy*dy <= cr*cr) flag = 1u;
        }

        prefixSumInput[tx] = flag;
        __syncthreads();

        sharedMemExclusiveScan(tx, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);

        if (tx == SCAN_BLOCK_DIM - 1) {
            tempCount = (int)(prefixSumOutput[tx] + prefixSumInput[tx]);
        }
        if (flag) {
            tileList[tileOffsets[b] + base + (int)prefixSumOutput[tx]] = c;
        }
        __syncthreads();

        base += tempCount;
        __syncthreads();

    }

    if (tx == 0) {
        tileCounts[b] = base;
    }
}


__global__ void kernelTilingRender(
    int tilesX, int tilesY,
    const int* __restrict__ tileOffsets,
    const int* __restrict__ tileCounts,
    const int* __restrict__ tileList,
    const int* __restrict__ tileActive,
    int isSnowflake)
{
    const int t = tileActive[blockIdx.x];
    const int bx = t % tilesX;
    const int by = t / tilesX;
    if (bx >= tilesX || by >= tilesY) return;

    const int imageW = cuConstRendererParams.imageWidth;
    const int imageH = cuConstRendererParams.imageHeight;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int px = bx * TILE_W + tx;
    const int py = by * TILE_H + ty;

    const bool active = (px < imageW) & (py < imageH);

    const int pixelIndex = 4 * (py * imageW + px);
    float4 img = make_float4(0.f, 0.f, 0.f, 0.f);
    if (active)
        img = *(float4*)(&cuConstRendererParams.imageData[pixelIndex]);

    const float2 pc = make_float2((px + 0.5f) / (float)imageW,
                                  (py + 0.5f) / (float)imageH);

    const int offset = tileOffsets[by * tilesX + bx];
    const int count = tileCounts[by * tilesX + bx];

    if (count <= 32) {
        if (px < imageW && py < imageH) {
            int pixIdx = 4 * (py * imageW + px);
            float4 img = *(float4*)(&cuConstRendererParams.imageData[pixIdx]);
            float2 pc = make_float2((px + 0.5f) / (float)imageW,
                                    (py + 0.5f) / (float)imageH);

            #pragma unroll
            for (int i = 0; i < count; ++i) {
                int c = tileList[offset + i];
                float3 p = *(float3*)(&cuConstRendererParams.position[3*c]);
                float  r =  cuConstRendererParams.radius[c];
                float dx = p.x - pc.x, dy = p.y - pc.y;
                float d2 = dx*dx + dy*dy;
                if (d2 > r*r) continue;

                float3 rgb; float a;
                if (isSnowflake) {
                    const float kA=.5f, falloff=4.f;
                    float norm = sqrtf(d2)/r;
                    rgb = lookupColor(norm);
                    float maxA = kA * fminf(fmaxf(.6f + .4f*(1.f-p.z), 0.f), 1.f);
                    a = maxA * expf(-falloff*norm*norm);
                } else {
                    rgb = *(float3*)(&cuConstRendererParams.color[3*c]);
                    a   = .5f;
                }
                float oma = 1.f - a;
                img.x = a*rgb.x + oma*img.x;
                img.y = a*rgb.y + oma*img.y;
                img.z = a*rgb.z + oma*img.z;
                img.w = a + img.w;
            }
            *(float4*)(&cuConstRendererParams.imageData[pixIdx]) = img;
        }
        return;   // skip shared-mem path
    }

    __shared__ float3 sPos[CIRCLES_PER_CHUNK];
    __shared__ float  sRad[CIRCLES_PER_CHUNK];
    __shared__ float3 sCol[CIRCLES_PER_CHUNK];

    const int tid = ty * TILE_W + tx;

    for (int i = 0; i < count; i += CIRCLES_PER_CHUNK) {
        const int n = min(CIRCLES_PER_CHUNK, count - i);

        if (tid < n) {
            const int c  = tileList[offset + i + tid];
            const int i3 = 3 * c;
            sPos[tid] = *(float3*)(&cuConstRendererParams.position[i3]);
            sRad[tid] =  cuConstRendererParams.radius[c];
            if (!isSnowflake)
                sCol[tid] = *(float3*)(&cuConstRendererParams.color[i3]);
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < n; j++) {
                const float dx = sPos[j].x - pc.x;
                const float dy = sPos[j].y - pc.y;
                const float d2 = dx*dx + dy*dy;
                const float r2 = sRad[j] * sRad[j];
                if (d2 > r2) continue;

                float3 rgb;
                float  alpha;

                if (isSnowflake) {
                    const float kCircleMaxAlpha = .5f;
                    const float falloff        = 4.f;
                    const float norm           = sqrtf(d2) / sRad[j];
                    rgb = lookupColor(norm);
                    float maxA = .6f + .4f * (1.f - sPos[j].z);
                    maxA = kCircleMaxAlpha * fminf(fmaxf(maxA, 0.f), 1.f);
                    alpha = maxA * expf(-falloff * norm * norm);
                } else {
                    rgb   = sCol[j];
                    alpha = .5f;
                }

                const float oneMinusAlpha = 1.f - alpha;
                img.x = alpha * rgb.x + oneMinusAlpha * img.x;
                img.y = alpha * rgb.y + oneMinusAlpha * img.y;
                img.z = alpha * rgb.z + oneMinusAlpha * img.z;
                img.w = alpha + img.w;
            }
        }
        __syncthreads();
    }

    if (active)
        *(float4*)(&cuConstRendererParams.imageData[pixelIndex]) = img;
}


__global__ void kernelRenderSmall(
    int imageW, int imageH, int N)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= imageW || py >= imageH) return;

    int pixelIndex = 4 * (py * imageW + px);
    float4 img = *(float4*)(&cuConstRendererParams.imageData[pixelIndex]);

    const float2 pc = make_float2((px + 0.5f) / (float)imageW,
                                  (py + 0.5f) / (float)imageH);

    for (int c = 0; c < N; c++) {
        const float3 p  = *(const float3*)(&cuConstRendererParams.position[3 * c]);
        const float  r  = cuConstRendererParams.radius[c];

        const float dx = p.x - pc.x;
        const float dy = p.y - pc.y;
        const float d2 = dx*dx + dy*dy;
        if (d2 > r*r) continue;

        float3 rgb = *(const float3*)(&cuConstRendererParams.color[3 * c]);
        float alpha = .5f;

        const float oneMinusAlpha = 1.f - alpha;
        img.x = alpha * rgb.x + oneMinusAlpha * img.x;
        img.y = alpha * rgb.y + oneMinusAlpha * img.y;
        img.z = alpha * rgb.z + oneMinusAlpha * img.z;
        img.w = alpha + img.w;
    }

    *(float4*)(&cuConstRendererParams.imageData[pixelIndex]) = img;
}


// mark tiles that have any circles
__global__ void kernelMarkActiveTiles(const int* __restrict__ tileCounts,
                                      int numTiles,
                                      int* __restrict__ flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numTiles) flags[i] = (tileCounts[i] > 0);
}

// write linear tile indices of active tiles using a scanned offset
__global__ void kernelBuildActiveTileList(const int* __restrict__ flags,
                                          const int* __restrict__ offsets,
                                          int numTiles,
                                          int* __restrict__ activeTiles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numTiles && flags[i]) {
        activeTiles[offsets[i]] = i;   // store linear tile id
    }
}

////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelScanBlocks(
    int* __restrict__ data,
    int N,
    int* __restrict__ blockSums)
{
    const int tid = threadIdx.x;
    const int base = blockIdx.x * SCAN_BLOCK_DIM;
    const int gid = base + tid;

    __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumScratch[2 * SCAN_BLOCK_DIM];

    uint value = (gid < N) ? (uint)data[gid] : 0u;
    prefixSumInput[tid] = value;
    __syncthreads();

    sharedMemExclusiveScan(tid, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);

    if (gid < N) {
        data[gid] = prefixSumOutput[tid];
    }

    if (tid == SCAN_BLOCK_DIM - 1) {
        blockSums[blockIdx.x] = (int)(prefixSumOutput[tid] + prefixSumInput[tid]);
    }
}

__global__ void kernelAddBlockSums(
    int* __restrict__ output,
    int N,
    const int* __restrict__ blockSumsOffset)
{
    const int tid = threadIdx.x;
    const int base = blockIdx.x * SCAN_BLOCK_DIM;
    const int gid = base + tid;
    const int offset = blockSumsOffset[blockIdx.x];

    if (gid < N) {
        output[gid] += offset;
    }
}

static inline void exclusive_scan(int* data, int length)
{
    if (length == 0) return;

    const int numBlocks = (length + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;

    int* blockSums = NULL;
    cudaMalloc(&blockSums, sizeof(int) * numBlocks);
    kernelScanBlocks<<<numBlocks, SCAN_BLOCK_DIM>>>(data, length, blockSums);

    if (numBlocks > 1) {
        exclusive_scan(blockSums, numBlocks);

        kernelAddBlockSums<<<numBlocks, SCAN_BLOCK_DIM>>>(data, length, blockSums);
    }
    cudaFree(blockSums);
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    tileX = 0;
    tileY = 0;
    numTiles = 0;
    tileLength = 0;

    d_tileCounts = NULL;
    d_tileOffsets = NULL;
    d_tileList = NULL;

    d_activeTileFlags = NULL;
    d_activeTileOffsets = NULL;
    d_activeTileList = NULL;

}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (d_tileCounts) cudaFree(d_tileCounts);
    if (d_tileOffsets) cudaFree(d_tileOffsets);
    if (d_tileList) cudaFree(d_tileList);

    if (d_activeTileFlags) cudaFree(d_activeTileFlags);
    if (d_activeTileOffsets) cudaFree(d_activeTileOffsets);
    if (d_activeTileList) cudaFree(d_activeTileList);

}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    tileX = (image->width + TILE_W - 1) / TILE_W;
    tileY = (image->height + TILE_H - 1) / TILE_H;
    numTiles = tileX * tileY;

    if (!d_tileCounts) cudaMalloc(&d_tileCounts, sizeof(int) * numTiles);
    if (!d_tileOffsets) cudaMalloc(&d_tileOffsets, sizeof(int) * numTiles);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}


void
CudaRenderer::render() {

    if (numberOfCircles <= 32) {
        dim3 block(32, 32);
        dim3 grid((image->width + block.x - 1) / block.x,
                  (image->height + block.y - 1) / block.y);
        kernelRenderSmall<<<grid, block>>>(image->width, image->height, numberOfCircles);
        cudaDeviceSynchronize();
        return;
    }

    // tile counts
    cudaMemset(d_tileCounts, 0, sizeof(int) * numTiles);
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x, 1, 1);
    kernelTileCounts<<<gridDim, blockDim>>>(tileX, tileY, d_tileCounts);
    cudaDeviceSynchronize();

    // exclusive scan
    cudaMemcpy(d_tileOffsets, d_tileCounts, sizeof(int) * numTiles, cudaMemcpyDeviceToDevice);
    exclusive_scan(d_tileOffsets, numTiles);
    cudaDeviceSynchronize();

    int last_offset=0;
    cudaMemcpy(&last_offset, &d_tileOffsets[numTiles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int last_count=0;
    cudaMemcpy(&last_count, &d_tileCounts[numTiles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int total_counts = last_offset + last_count;

    if (total_counts > tileLength) {
        if (d_tileList) cudaFree(d_tileList);
        cudaMalloc(&d_tileList, sizeof(int) * total_counts);
        tileLength = total_counts;
    }

    // soreted tile list
    dim3 blockDim2(numTiles, 1, 1);
    kernelSortedTileList<<<numTiles, SCAN_BLOCK_DIM>>>(tileX, tileY, numberOfCircles, d_tileOffsets, d_tileCounts, d_tileList);
    cudaDeviceSynchronize();

    cudaMalloc(&d_activeTileFlags, sizeof(int) * numTiles);
    cudaMalloc(&d_activeTileOffsets, sizeof(int) * numTiles);

    dim3 blockDim3(256, 1, 1);
    dim3 gridDim3((numTiles + blockDim3.x - 1) / blockDim3.x, 1, 1);
    kernelMarkActiveTiles<<<gridDim3, blockDim3>>>(d_tileCounts, numTiles, d_activeTileFlags);
    cudaDeviceSynchronize();

    cudaMemcpy(d_activeTileOffsets, d_activeTileFlags, sizeof(int) * numTiles, cudaMemcpyDeviceToDevice);
    exclusive_scan(d_activeTileOffsets, numTiles);
    cudaDeviceSynchronize();

    int last_active_offset = 0;
    cudaMemcpy(&last_active_offset, &d_activeTileOffsets[numTiles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int last_active_flag = 0;
    cudaMemcpy(&last_active_flag, &d_activeTileFlags[numTiles - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int total_active_tiles = last_active_offset + last_active_flag;

    cudaMalloc(&d_activeTileList, sizeof(int) * total_active_tiles);
    kernelBuildActiveTileList<<<gridDim3, blockDim3>>>(d_activeTileFlags, d_activeTileOffsets, numTiles, d_activeTileList);
    cudaDeviceSynchronize();

    dim3 blockDim4(TILE_W, TILE_H, 1);
    dim3 gridDim4(total_active_tiles, 1, 1);
    bool isSnowflake = (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME);
    kernelTilingRender<<<gridDim4, blockDim4>>>(tileX, tileY, d_tileOffsets, d_tileCounts, d_tileList, d_activeTileList,isSnowflake);
    cudaDeviceSynchronize();
}
