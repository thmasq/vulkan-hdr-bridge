// Build: g++ main.cpp -o hdr_demo -lvulkan -lglfw -lshaderc_shared -std=c++17 -O3

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <shaderc/shaderc.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

static const char *SRC_SCENE_VERT = R"glsl(
#version 450
layout(location = 0) out vec3 col;

layout(push_constant) uniform Push {
    mat4 rotation;
} pc;

vec2 pos[3] = vec2[](vec2(-0.8,0.8), vec2(0.8,0.8), vec2(0.0,-0.8));
vec3 clr[3] = vec3[](
    vec3(8.0,  0.05, 0.05),
    vec3(0.05, 0.05, 10.0),
    vec3(0.05, 9.0,  0.05)
);

void main() {
    gl_Position = pc.rotation * vec4(pos[gl_VertexIndex], 0.0, 1.0);
    col = clr[gl_VertexIndex];
}
)glsl";

static const char *SRC_SCENE_FRAG = R"glsl(
#version 450
layout(location = 0) in  vec3 col;
layout(location = 0) out vec4 outCol;
void main() { outCol = vec4(col, 1.0); }
)glsl";

static const char *SRC_TONEMAP_VERT = R"glsl(
#version 450
layout(location = 0) out vec2 uv;
void main() {
    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

static const char *SRC_TONEMAP_FRAG = R"glsl(
#version 450
layout(location = 0) in  vec2 uv;
layout(location = 0) out vec4 outCol;

layout(binding = 0) uniform sampler2DMS hdrTex;
layout(push_constant) uniform PC { int hdr; } pc;

vec3 st2084_oetf(vec3 c) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 cp = pow(c, vec3(m1));
    return pow((c1 + c2 * cp) / (1.0 + c3 * cp), vec3(m2));
}

vec3 rec709_to_rec2020(vec3 c) {
    mat3 m = mat3(
        0.6274040, 0.0690970, 0.0163916,
        0.3292820, 0.9195400, 0.0880132,
        0.0433136, 0.0113612, 0.8955950
    );
    return m * c;
}

vec3 aces(vec3 x) {
    const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    ivec2 texSize = textureSize(hdrTex);
    ivec2 texCoord = ivec2(uv * vec2(texSize));
    
    vec3 accumulatedColor = vec3(0.0);
    int numSamples = 4;
    
    const float paperWhiteNits = 204.0; 
    const float pqMaxNits = 10000.0; 
    const float displayPeakNits = 1000.0; 
    const float kneeStart = 700.0;
    
    for (int i = 0; i < numSamples; i++) {
        vec3 h = texelFetch(hdrTex, texCoord, i).rgb;
        vec3 colorNits;
        
        if (pc.hdr == 1) {
            colorNits = h * paperWhiteNits; 
            float maxChannel = max(max(colorNits.r, colorNits.g), colorNits.b);
            
            if (maxChannel > kneeStart) {
                float t = maxChannel - kneeStart;
                float rollOffMax = displayPeakNits - kneeStart;
                float newMax = kneeStart + rollOffMax * (t / (rollOffMax + t));
                colorNits = colorNits * (newMax / maxChannel);
            }
        } else {
            float maxChannel = max(max(h.r, h.g), h.b);
            
            if (maxChannel > 0.0001) {
                float mappedMax = aces(vec3(maxChannel * 0.6)).x;
                vec3 sdr = h * (mappedMax / maxChannel);
                colorNits = sdr * paperWhiteNits;
            } else {
                colorNits = vec3(0.0);
            }
        }
        
        vec3 rec2020 = rec709_to_rec2020(colorNits);
        vec3 pq = st2084_oetf(clamp(rec2020 / pqMaxNits, 0.0, 1.0));
        
        accumulatedColor += pq;
    }
    
    outCol = vec4(accumulatedColor / float(numSamples), 1.0);
}
)glsl";

static bool g_hdrMode = true;

static std::vector<uint32_t> compileGLSL(const char *src,
                                         shaderc_shader_kind kind)
{
	shaderc::Compiler c;
	shaderc::CompileOptions opts;
	opts.SetOptimizationLevel(shaderc_optimization_level_performance);
	auto res = c.CompileGlslToSpv(src, strlen(src), kind, "shader", opts);
	if (res.GetCompilationStatus() != shaderc_compilation_status_success)
		throw std::runtime_error(res.GetErrorMessage());
	return {res.cbegin(), res.cend()};
}

static void keyCallback(GLFWwindow *w, int key, int, int action, int)
{
	if (key == GLFW_KEY_H && action == GLFW_PRESS) {
		g_hdrMode = !g_hdrMode;
		std::cout << "Mode: "
		          << (g_hdrMode ? "True HDR (ST2084)"
		                        : "Simulated SDR (ACES -> 204 nits)")
		          << "\n";
	}
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(w, GLFW_TRUE);
}

class App
{
	static constexpr int W = 1280, H = 720, MFIF = 2;
	static constexpr VkFormat HDR_FMT = VK_FORMAT_R16G16B16A16_SFLOAT;

	GLFWwindow *wnd{};
	VkInstance inst{};
	VkSurfaceKHR surf{};
	VkPhysicalDevice pdev{};
	VkDevice dev{};
	VkQueue gfxQ{}, presQ{};
	uint32_t gfxFam{}, presFam{};
	VkSwapchainKHR swapchain{};
	std::vector<VkImage> scImages;
	std::vector<VkImageView> scViews;
	VkFormat scFmt{};
	VkExtent2D scExt{};

	VkImage msaaImg{};
	VkDeviceMemory msaaMem{};
	VkImageView msaaView{};

	VkSampler hdrSampler{};

	VkRenderPass sceneRP{};
	VkRenderPass tonemapRP{};

	VkFramebuffer sceneFB{};
	std::vector<VkFramebuffer> tonemapFBs;

	VkDescriptorSetLayout dsl{};
	VkDescriptorPool dpool{};
	VkDescriptorSet dset{};

	VkPipelineLayout scenePL{};
	VkPipeline scenePipe{};
	VkPipelineLayout tonemapPL{};
	VkPipeline tonemapPipe{};

	VkCommandPool cmdPool{};
	std::vector<VkCommandBuffer> cmdBufs;

	std::vector<VkSemaphore> imgAvail;
	std::vector<VkSemaphore> renderDone;
	std::vector<VkFence> inFlight;
	int frame{0};

	uint32_t findMem(uint32_t filter, VkMemoryPropertyFlags props)
	{
		VkPhysicalDeviceMemoryProperties mp;
		vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
		for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
			if ((filter & (1u << i)) &&
			    (mp.memoryTypes[i].propertyFlags & props) == props)
				return i;
		throw std::runtime_error("No suitable memory type");
	}

	VkShaderModule mkShader(const std::vector<uint32_t> &spv)
	{
		VkShaderModuleCreateInfo ci{
		    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		ci.codeSize = spv.size() * 4;
		ci.pCode = spv.data();
		VkShaderModule m;
		vkCreateShaderModule(dev, &ci, nullptr, &m);
		return m;
	}

	VkImageView mkImageView(VkImage img, VkFormat fmt)
	{
		VkImageViewCreateInfo ci{
		    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		ci.image = img;
		ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
		ci.format = fmt;
		ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		VkImageView v;
		vkCreateImageView(dev, &ci, nullptr, &v);
		return v;
	}

	void mkInstance()
	{
		VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
		ai.pApplicationName = "HDRDemo";
		ai.apiVersion = VK_API_VERSION_1_3;

		uint32_t gc;
		const char **ge = glfwGetRequiredInstanceExtensions(&gc);
		std::vector<const char *> exts(ge, ge + gc);

		exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

		const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
		VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ci.pApplicationInfo = &ai;
		ci.enabledExtensionCount = (uint32_t)exts.size();
		ci.ppEnabledExtensionNames = exts.data();
		ci.enabledLayerCount = 1;
		ci.ppEnabledLayerNames = layers;

		if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS)
			throw std::runtime_error("vkCreateInstance failed");
	}

	void pickPhysicalDevice()
	{
		uint32_t n;
		vkEnumeratePhysicalDevices(inst, &n, nullptr);
		if (!n)
			throw std::runtime_error("No Vulkan devices");
		std::vector<VkPhysicalDevice> devs(n);
		vkEnumeratePhysicalDevices(inst, &n, devs.data());
		pdev = devs[0];
	}

	void findQueues()
	{
		uint32_t n;
		vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n, nullptr);
		std::vector<VkQueueFamilyProperties> qf(n);
		vkGetPhysicalDeviceQueueFamilyProperties(pdev, &n, qf.data());

		gfxFam = UINT32_MAX;
		presFam = UINT32_MAX;

		for (uint32_t i = 0; i < n; i++) {
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(pdev, i, surf, &presentSupport);

			if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				gfxFam = i;

			if (presentSupport)
				presFam = i;

			if (gfxFam != UINT32_MAX && gfxFam == presFam) {
				break;
			}
		}
	
		if (gfxFam == UINT32_MAX || presFam == UINT32_MAX)
			throw std::runtime_error("Failed to find suitable queue families");
	}

	void mkDevice()
	{
		std::set<uint32_t> fams{gfxFam, presFam};
		std::vector<VkDeviceQueueCreateInfo> qci;
		float pri = 1.f;
		for (auto f : fams) {
			VkDeviceQueueCreateInfo q{
			    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
			q.queueFamilyIndex = f;
			q.queueCount = 1;
			q.pQueuePriorities = &pri;
			qci.push_back(q);
		}

		const char *exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		                      VK_EXT_HDR_METADATA_EXTENSION_NAME};

		VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		ci.queueCreateInfoCount = (uint32_t)qci.size();
		ci.pQueueCreateInfos = qci.data();
		ci.enabledExtensionCount = 2;
		ci.ppEnabledExtensionNames = exts;

		if (vkCreateDevice(pdev, &ci, nullptr, &dev) != VK_SUCCESS)
			throw std::runtime_error("vkCreateDevice failed");
		vkGetDeviceQueue(dev, gfxFam, 0, &gfxQ);
		vkGetDeviceQueue(dev, presFam, 0, &presQ);
	}

	void mkSwapchain()
	{
		VkSurfaceCapabilitiesKHR caps;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdev, surf, &caps);

		if (caps.currentExtent.width != UINT32_MAX) {
			scExt = caps.currentExtent;
		} else {
			int w, h;
			glfwGetFramebufferSize(wnd, &w, &h);
			scExt.width =
			    std::clamp((uint32_t)w, caps.minImageExtent.width,
			               caps.maxImageExtent.width);
			scExt.height =
			    std::clamp((uint32_t)h, caps.minImageExtent.height,
			               caps.maxImageExtent.height);
		}

		uint32_t n;
		vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surf, &n, nullptr);
		std::vector<VkSurfaceFormatKHR> fmts(n);
		vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surf, &n,
		                                     fmts.data());

		VkSurfaceFormatKHR chosen = fmts[0];
		bool hdrSupport = false;
		for (auto &f : fmts) {
			if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
			    f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
				chosen = f;
				hdrSupport = true;
				break;
			}
		}

		if (!hdrSupport) {
			std::cout << "\n[WARNING] True HDR10 surface format "
			             "not found.\n";
			std::cout << "Falling back to standard SDR format.\n\n";
		}

		scFmt = chosen.format;

		uint32_t imgCount = std::min(
		    caps.minImageCount + 1,
		    caps.maxImageCount ? caps.maxImageCount : UINT32_MAX);
		VkSwapchainCreateInfoKHR ci{
		    VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
		ci.surface = surf;
		ci.minImageCount = imgCount;
		ci.imageFormat = scFmt;
		ci.imageColorSpace = chosen.colorSpace;
		ci.imageExtent = scExt;
		ci.imageArrayLayers = 1;
		ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		uint32_t fam[2] = {gfxFam, presFam};
		if (gfxFam != presFam) {
			ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			ci.queueFamilyIndexCount = 2;
			ci.pQueueFamilyIndices = fam;
		} else {
			ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}
		ci.preTransform = caps.currentTransform;
		ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
		ci.clipped = VK_TRUE;
		vkCreateSwapchainKHR(dev, &ci, nullptr, &swapchain);

		vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr);
		scImages.resize(n);
		vkGetSwapchainImagesKHR(dev, swapchain, &n, scImages.data());
		scViews.reserve(n);
		for (auto img : scImages)
			scViews.push_back(mkImageView(img, scFmt));
	}

	void setHdrMetadata()
	{
		auto pfnSetHdrMetadataEXT =
		    (PFN_vkSetHdrMetadataEXT)vkGetDeviceProcAddr(
		        dev, "vkSetHdrMetadataEXT");
		if (!pfnSetHdrMetadataEXT) {
			std::cerr
			    << "Warning: vkSetHdrMetadataEXT not found.\n";
			return;
		}

		VkHdrMetadataEXT metadata{VK_STRUCTURE_TYPE_HDR_METADATA_EXT};
		metadata.displayPrimaryRed = {0.708f, 0.292f};
		metadata.displayPrimaryGreen = {0.170f, 0.797f};
		metadata.displayPrimaryBlue = {0.131f, 0.046f};
		metadata.whitePoint = {0.3127f, 0.3290f};
		metadata.maxLuminance = 1000.0f;
		metadata.minLuminance = 0.0f;
		metadata.maxContentLightLevel = 1000.0f;
		metadata.maxFrameAverageLightLevel = 400.0f;

		pfnSetHdrMetadataEXT(dev, 1, &swapchain, &metadata);
	}

	void mkHDRBuffer()
	{
		VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		ci.imageType = VK_IMAGE_TYPE_2D;
		ci.format = HDR_FMT;
		ci.extent = {scExt.width, scExt.height, 1};
		ci.mipLevels = 1;
		ci.arrayLayers = 1;
		ci.samples = VK_SAMPLE_COUNT_4_BIT;
		ci.tiling = VK_IMAGE_TILING_OPTIMAL;
		ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		           VK_IMAGE_USAGE_SAMPLED_BIT;
		ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		vkCreateImage(dev, &ci, nullptr, &msaaImg);

		VkMemoryRequirements mr;
		vkGetImageMemoryRequirements(dev, msaaImg, &mr);
		VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		ai.allocationSize = mr.size;
		ai.memoryTypeIndex = findMem(
		    mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(dev, &ai, nullptr, &msaaMem);
		vkBindImageMemory(dev, msaaImg, msaaMem, 0);
		msaaView = mkImageView(msaaImg, HDR_FMT);

		VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		si.magFilter = si.minFilter = VK_FILTER_LINEAR;
		si.addressModeU = si.addressModeV = si.addressModeW =
		    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		vkCreateSampler(dev, &si, nullptr, &hdrSampler);
	}

	void mkRenderPasses()
	{
		{
			VkAttachmentDescription atts[1]{};
			atts[0].format = HDR_FMT;
			atts[0].samples = VK_SAMPLE_COUNT_4_BIT;
			atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			atts[0].finalLayout =
			    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkAttachmentReference colorRef{
			    0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

			VkSubpassDescription sub{};
			sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			sub.colorAttachmentCount = 1;
			sub.pColorAttachments = &colorRef;
			sub.pResolveAttachments = nullptr;

			VkSubpassDependency dep{};
			dep.srcSubpass = VK_SUBPASS_EXTERNAL;
			dep.dstSubpass = 0;
			dep.srcStageMask =
			    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			dep.dstStageMask =
			    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dep.dstAccessMask =
			    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo ci{
			    VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
			ci.attachmentCount = 1;
			ci.pAttachments = atts;
			ci.subpassCount = 1;
			ci.pSubpasses = &sub;
			ci.dependencyCount = 1;
			ci.pDependencies = &dep;
			vkCreateRenderPass(dev, &ci, nullptr, &sceneRP);
		}
		{
			VkAttachmentDescription att{};
			att.format = scFmt;
			att.samples = VK_SAMPLE_COUNT_1_BIT;
			att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			VkAttachmentReference ref{
			    0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
			VkSubpassDescription sub{};
			sub.colorAttachmentCount = 1;
			sub.pColorAttachments = &ref;
			VkSubpassDependency dep{};
			dep.srcSubpass = VK_SUBPASS_EXTERNAL;
			dep.dstSubpass = 0;
			dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			VkRenderPassCreateInfo ci{
			    VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
			ci.attachmentCount = 1;
			ci.pAttachments = &att;
			ci.subpassCount = 1;
			ci.pSubpasses = &sub;
			ci.dependencyCount = 1;
			ci.pDependencies = &dep;
			vkCreateRenderPass(dev, &ci, nullptr, &tonemapRP);
		}
	}

	void mkFramebuffers()
	{
		{
			VkImageView fbAtts[1] = {msaaView};
			VkFramebufferCreateInfo ci{
			    VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
			ci.renderPass = sceneRP;
			ci.attachmentCount = 1;
			ci.pAttachments = fbAtts;
			ci.width = scExt.width;
			ci.height = scExt.height;
			ci.layers = 1;
			vkCreateFramebuffer(dev, &ci, nullptr, &sceneFB);
		}
		tonemapFBs.resize(scViews.size());
		for (size_t i = 0; i < scViews.size(); i++) {
			VkFramebufferCreateInfo ci{
			    VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
			ci.renderPass = tonemapRP;
			ci.attachmentCount = 1;
			ci.pAttachments = &scViews[i];
			ci.width = scExt.width;
			ci.height = scExt.height;
			ci.layers = 1;
			vkCreateFramebuffer(dev, &ci, nullptr, &tonemapFBs[i]);
		}
	}

	void mkDescriptors()
	{
		VkDescriptorSetLayoutBinding b{};
		b.binding = 0;
		b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b.descriptorCount = 1;
		b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		VkDescriptorSetLayoutCreateInfo lci{
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		lci.bindingCount = 1;
		lci.pBindings = &b;
		vkCreateDescriptorSetLayout(dev, &lci, nullptr, &dsl);

		VkDescriptorPoolSize ps{
		    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
		VkDescriptorPoolCreateInfo pci{
		    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pci.maxSets = 1;
		pci.poolSizeCount = 1;
		pci.pPoolSizes = &ps;
		vkCreateDescriptorPool(dev, &pci, nullptr, &dpool);

		VkDescriptorSetAllocateInfo ai{
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		ai.descriptorPool = dpool;
		ai.descriptorSetCount = 1;
		ai.pSetLayouts = &dsl;
		vkAllocateDescriptorSets(dev, &ai, &dset);

		VkDescriptorImageInfo ii{
		    hdrSampler, msaaView,
		    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wr.dstSet = dset;
		wr.dstBinding = 0;
		wr.descriptorCount = 1;
		wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		wr.pImageInfo = &ii;
		vkUpdateDescriptorSets(dev, 1, &wr, 0, nullptr);
	}

	VkPipeline mkPipeline(VkShaderModule vs, VkShaderModule fs,
	                      VkRenderPass rp, VkPipelineLayout pl,
	                      VkSampleCountFlagBits samples)
	{
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType =
		    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vs;
		stages[0].pName = "main";
		stages[1].sType =
		    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fs;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo vi{
		    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		VkPipelineInputAssemblyStateCreateInfo ia{
		    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		VkViewport vp{0, 0, (float)scExt.width, (float)scExt.height,
		              0, 1};
		VkRect2D sc{{0, 0}, scExt};
		VkPipelineViewportStateCreateInfo vps{
		    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		vps.viewportCount = 1;
		vps.pViewports = &vp;
		vps.scissorCount = 1;
		vps.pScissors = &sc;

		VkPipelineRasterizationStateCreateInfo rs{
		    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		rs.polygonMode = VK_POLYGON_MODE_FILL;
		rs.cullMode = VK_CULL_MODE_NONE;
		rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rs.lineWidth = 1;

		VkPipelineMultisampleStateCreateInfo ms{
		    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		ms.rasterizationSamples = samples;

		VkPipelineColorBlendAttachmentState cba{};
		cba.colorWriteMask = 0xF;
		VkPipelineColorBlendStateCreateInfo cb{
		    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		cb.attachmentCount = 1;
		cb.pAttachments = &cba;

		VkGraphicsPipelineCreateInfo ci{
		    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		ci.stageCount = 2;
		ci.pStages = stages;
		ci.pVertexInputState = &vi;
		ci.pInputAssemblyState = &ia;
		ci.pViewportState = &vps;
		ci.pRasterizationState = &rs;
		ci.pMultisampleState = &ms;
		ci.pColorBlendState = &cb;
		ci.layout = pl;
		ci.renderPass = rp;
		ci.subpass = 0;

		VkPipeline pipe;
		vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
		                          &pipe);
		return pipe;
	}

	void mkPipelines()
	{
		{
			VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0,
			                        sizeof(glm::mat4)};
			VkPipelineLayoutCreateInfo ci{
			    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
			ci.pushConstantRangeCount = 1;
			ci.pPushConstantRanges = &pcr;
			vkCreatePipelineLayout(dev, &ci, nullptr, &scenePL);
		}
		{
			VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0,
			                        sizeof(int)};
			VkPipelineLayoutCreateInfo ci{
			    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
			ci.setLayoutCount = 1;
			ci.pSetLayouts = &dsl;
			ci.pushConstantRangeCount = 1;
			ci.pPushConstantRanges = &pcr;
			vkCreatePipelineLayout(dev, &ci, nullptr, &tonemapPL);
		}

		auto spvSV =
		    compileGLSL(SRC_SCENE_VERT, shaderc_glsl_vertex_shader);
		auto spvSF =
		    compileGLSL(SRC_SCENE_FRAG, shaderc_glsl_fragment_shader);
		auto spvTV =
		    compileGLSL(SRC_TONEMAP_VERT, shaderc_glsl_vertex_shader);
		auto spvTF =
		    compileGLSL(SRC_TONEMAP_FRAG, shaderc_glsl_fragment_shader);

		auto sv = mkShader(spvSV), sf = mkShader(spvSF);
		auto tv = mkShader(spvTV), tf = mkShader(spvTF);

		scenePipe =
		    mkPipeline(sv, sf, sceneRP, scenePL, VK_SAMPLE_COUNT_4_BIT);
		tonemapPipe = mkPipeline(tv, tf, tonemapRP, tonemapPL,
		                         VK_SAMPLE_COUNT_1_BIT);

		vkDestroyShaderModule(dev, sv, nullptr);
		vkDestroyShaderModule(dev, sf, nullptr);
		vkDestroyShaderModule(dev, tv, nullptr);
		vkDestroyShaderModule(dev, tf, nullptr);
	}

	void mkCommandBuffers()
	{
		VkCommandPoolCreateInfo ci{
		    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		ci.queueFamilyIndex = gfxFam;
		ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		vkCreateCommandPool(dev, &ci, nullptr, &cmdPool);

		cmdBufs.resize(MFIF);
		VkCommandBufferAllocateInfo ai{
		    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		ai.commandPool = cmdPool;
		ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ai.commandBufferCount = MFIF;
		vkAllocateCommandBuffers(dev, &ai, cmdBufs.data());
	}

	void mkSync()
	{
		imgAvail.resize(MFIF);
		inFlight.resize(MFIF);
		renderDone.resize(scImages.size());

		VkSemaphoreCreateInfo si{
		    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (int i = 0; i < MFIF; i++) {
			vkCreateSemaphore(dev, &si, nullptr, &imgAvail[i]);
			vkCreateFence(dev, &fi, nullptr, &inFlight[i]);
		}
		for (size_t i = 0; i < scImages.size(); i++) {
			vkCreateSemaphore(dev, &si, nullptr, &renderDone[i]);
		}
	}

	void drawFrame()
	{
		vkWaitForFences(dev, 1, &inFlight[frame], VK_TRUE, UINT64_MAX);

		uint32_t imgIdx;
		vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX,
		                      imgAvail[frame], VK_NULL_HANDLE, &imgIdx);

		vkResetFences(dev, 1, &inFlight[frame]);

		auto cmd = cmdBufs[frame];
		vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo bi{
		    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		vkBeginCommandBuffer(cmd, &bi);

		float time = glfwGetTime();
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), time,
		                                 glm::vec3(0.0f, 0.0f, 1.0f));

		VkClearValue clrs[1]{};
		clrs[0].color.float32[0] = 0.0f;
		clrs[0].color.float32[1] = 0.0f;
		clrs[0].color.float32[2] = 0.0f;
		clrs[0].color.float32[3] = 1.0f;

		VkRect2D area{{0, 0}, scExt};

		{
			VkRenderPassBeginInfo rp{
			    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
			rp.renderPass = sceneRP;
			rp.framebuffer = sceneFB;
			rp.renderArea = area;
			rp.clearValueCount = 1;
			rp.pClearValues = clrs;
			vkCmdBeginRenderPass(cmd, &rp,
			                     VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                  scenePipe);

			vkCmdPushConstants(cmd, scenePL,
			                   VK_SHADER_STAGE_VERTEX_BIT, 0,
			                   sizeof(rotation), &rotation);

			vkCmdDraw(cmd, 3, 1, 0, 0);
			vkCmdEndRenderPass(cmd);
		}

		{
			VkRenderPassBeginInfo rp{
			    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
			rp.renderPass = tonemapRP;
			rp.framebuffer = tonemapFBs[imgIdx];
			rp.renderArea = area;
			rp.clearValueCount = 0;
			vkCmdBeginRenderPass(cmd, &rp,
			                     VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
			                  tonemapPipe);
			vkCmdBindDescriptorSets(
			    cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPL, 0,
			    1, &dset, 0, nullptr);
			int hdrFlag = g_hdrMode ? 1 : 0;
			vkCmdPushConstants(cmd, tonemapPL,
			                   VK_SHADER_STAGE_FRAGMENT_BIT, 0,
			                   sizeof(int), &hdrFlag);
			vkCmdDraw(cmd, 3, 1, 0, 0);
			vkCmdEndRenderPass(cmd);
		}

		vkEndCommandBuffer(cmd);

		VkPipelineStageFlags wait =
		    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		sub.waitSemaphoreCount = 1;
		sub.pWaitSemaphores = &imgAvail[frame];
		sub.pWaitDstStageMask = &wait;
		sub.commandBufferCount = 1;
		sub.pCommandBuffers = &cmd;
		sub.signalSemaphoreCount = 1;
		sub.pSignalSemaphores = &renderDone[imgIdx];
		vkQueueSubmit(gfxQ, 1, &sub, inFlight[frame]);

		VkPresentInfoKHR pr{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
		pr.waitSemaphoreCount = 1;
		pr.pWaitSemaphores = &renderDone[imgIdx];
		pr.swapchainCount = 1;
		pr.pSwapchains = &swapchain;
		pr.pImageIndices = &imgIdx;
		vkQueuePresentKHR(presQ, &pr);

		frame = (frame + 1) % MFIF;
	}

      public:
	void run()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		wnd = glfwCreateWindow(W, H, "Vulkan True HDR10 Demo", nullptr,
		                       nullptr);
		glfwSetKeyCallback(wnd, keyCallback);

		mkInstance();
		glfwCreateWindowSurface(inst, wnd, nullptr, &surf);
		pickPhysicalDevice();
		findQueues();
		mkDevice();
		mkSwapchain();

		setHdrMetadata();

		mkHDRBuffer();
		mkRenderPasses();
		mkFramebuffers();
		mkDescriptors();
		mkPipelines();
		mkCommandBuffers();
		mkSync();

		std::cout << "Press H to toggle True HDR / Simulated SDR.\n";

		while (!glfwWindowShouldClose(wnd)) {
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(dev);
	}

~App()
	{
		if (dev) {
			for (auto s : imgAvail)
				vkDestroySemaphore(dev, s, nullptr);
			for (auto s : renderDone)
				vkDestroySemaphore(dev, s, nullptr);
			for (auto f : inFlight)
				vkDestroyFence(dev, f, nullptr);
			
			vkDestroyCommandPool(dev, cmdPool, nullptr);
			vkDestroyPipeline(dev, scenePipe, nullptr);
			vkDestroyPipelineLayout(dev, scenePL, nullptr);
			vkDestroyPipeline(dev, tonemapPipe, nullptr);
			vkDestroyPipelineLayout(dev, tonemapPL, nullptr);
			vkDestroyDescriptorPool(dev, dpool, nullptr);
			vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
			
			for (auto fb : tonemapFBs)
				vkDestroyFramebuffer(dev, fb, nullptr);
			vkDestroyFramebuffer(dev, sceneFB, nullptr);
			vkDestroyRenderPass(dev, sceneRP, nullptr);
			vkDestroyRenderPass(dev, tonemapRP, nullptr);
			vkDestroySampler(dev, hdrSampler, nullptr);

			vkDestroyImageView(dev, msaaView, nullptr);
			vkDestroyImage(dev, msaaImg, nullptr);
			vkFreeMemory(dev, msaaMem, nullptr);

			for (auto v : scViews)
				vkDestroyImageView(dev, v, nullptr);
				
			vkDestroySwapchainKHR(dev, swapchain, nullptr);
			vkDestroyDevice(dev, nullptr);
		}
		
		if (inst) {
			vkDestroySurfaceKHR(inst, surf, nullptr);
			vkDestroyInstance(inst, nullptr);
		}
		
		if (wnd) {
			glfwDestroyWindow(wnd);
			glfwTerminate();
		}
	}
};

int main()
{
	try {
		App{}.run();
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\n";
		return 1;
	}
}
