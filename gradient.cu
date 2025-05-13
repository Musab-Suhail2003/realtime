#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 600
#define GRAVITY 9.81

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float len = sqrtf(dot(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ bool hitSphere(float3 center, float radius, float3 rayOrigin, float3 rayDir, float* t) {
    float3 oc = make_float3(rayOrigin.x - center.x, rayOrigin.y - center.y, rayOrigin.z - center.z);
    float a = dot(rayDir, rayDir);
    float b = 2.0f * dot(oc, rayDir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    } else {
        *t = (-b - sqrtf(discriminant)) / (2.0f * a);
        return *t > 0.0f;
    }
}

__global__ void render(uchar4* pixels, float sphereY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float u = (2.0f * x) / (float)(WIDTH - 1) - 1.0f;
    float v = (2.0f * y) / (float)(HEIGHT - 1) - 1.0f;
    float aspect = WIDTH / (float)HEIGHT;

    float3 rayOrigin = make_float3(0.0f, 0.0f, 0.0f);
    float3 rayDir = make_float3(u * aspect, v, -1.0f);
    rayDir = normalize(rayDir);

    float t;
    float3 sphereCenter = make_float3(0.0f, sphereY, -1.0f);
    float radius = 0.1f;

    unsigned char r, g, b;

    if (hitSphere(sphereCenter, radius, rayOrigin, rayDir, &t)) {
        float3 hitPoint = make_float3(rayOrigin.x + t * rayDir.x,
                                     rayOrigin.y + t * rayDir.y,
                                     rayOrigin.z + t * rayDir.z);

        float3 normal = make_float3((hitPoint.x - sphereCenter.x) / radius,
                                    (hitPoint.y - sphereCenter.y) / radius,
                                    (hitPoint.z - sphereCenter.z) / radius);

        float3 lightDir = make_float3(1.0f, 1.0f, -1.0f);
        lightDir = normalize(lightDir);

        float diff = fmaxf(dot(normal, lightDir), 0.0f);
        diff = sqrtf(diff);

        r = (unsigned char)(diff * 255);
        g = (unsigned char)(diff * 0.5f * 255);
        b = (unsigned char)(diff * 0.7f * 255);
    } else {
        float t = 0.5f * (v + 1.0f);
        r = (unsigned char)((1.0f * (1.0f - t) + 0.5f * t) * 255);
        g = (unsigned char)((1.0f * (1.0f - t) + 0.7f * t) * 255);
        b = (unsigned char)((1.0f * (1.0f - t) + 1.0f * t) * 255);
    }

    int idx = y * WIDTH + x;
    pixels[idx] = make_uchar4(r, g, b, 255);
}

GLuint createTexture() {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

int main(int argc, char** argv) {
    float time = 0.0f;
    float initialY = 1.0f; 
    float velocity = 0.0f;
    float damping = 0.95f; // Damping factor for energy loss on bounce
    float mass = 1.0f; // Mass of the sphere (in kg)
    float collisionTime = 0.01f; // Approximate collision duration (in seconds)

    if (argc > 1) {
        damping = atof(argv[1]); // Set initial position from command line argument
    } else {
        printf("No damping factor provided, using default value of 0.95\n");
    }

    if (!glfwInit()) {
        printf("Failed to init GLFW!\n");
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA + OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);

    glewInit();

    GLuint tex = createTexture();

    cudaGraphicsResource* cudaTexResource;
    cudaGraphicsGLRegisterImage(&cudaTexResource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    while (!glfwWindowShouldClose(window)) {
        time += 0.016f; // ~60fps

        // Update sphere's position using gravity
        float sphereY = initialY + velocity * time - 0.5f * GRAVITY * time * time;

        // Check if the sphere hits the bottom boundary
        if (sphereY < -0.865f) {
            sphereY = -0.865f; // Clamp position to the bottom boundary

            // Ensure collisionTime and mass are valid
            if (collisionTime <= 0.0f) collisionTime = 0.01f; // Prevent division by zero
            if (mass <= 0.0f) mass = 1.0f; // Prevent division by zero

            // Calculate the normal force
            float normalForce = (mass * fabs(velocity)) / collisionTime;
           
            // Calculate the reverse velocity based on damping
            float reverseVelocity = -velocity * damping;

            // Update the sphere's velocity
            velocity = reverseVelocity;

            // Reset initial position and time
            initialY = sphereY;
            time = 0.0f;
        }

        // Check if the sphere hits the top boundary
        if (sphereY > 1.0f) {
            sphereY = 1.0f; // Clamp position to the top boundary
            velocity = -velocity * damping; // Reverse and dampen velocity
            initialY = sphereY; // Reset initial position
            time = 0.0f; // Reset time
        }

        // Update velocity for the next frame
        velocity -= GRAVITY * 0.016f;

        cudaArray_t array;
        cudaGraphicsMapResources(1, &cudaTexResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&array, cudaTexResource, 0, 0);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        uchar4* devPtr;
        size_t size;
        cudaMallocPitch((void**)&devPtr, &size, WIDTH * sizeof(uchar4), HEIGHT);

        render<<<gridSize, blockSize>>>(devPtr, sphereY);

        cudaMemcpy2DToArray(array, 0, 0, devPtr, WIDTH * sizeof(uchar4), WIDTH * sizeof(uchar4), HEIGHT, cudaMemcpyDeviceToDevice);

        cudaFree(devPtr);
        cudaGraphicsUnmapResources(1, &cudaTexResource, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaTexResource);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

