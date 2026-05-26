#pragma once
#include <Renderer/scene-proxy.h>
#include <Renderer/frame-queue.h>
#include <memory>
#include <thread>
#include <functional>
#include <atomic>

namespace sim::renderer {

/// 渲染器配置
struct RendererConfig {
  int windowWidth = 1280;
  int windowHeight = 720;
  std::string windowTitle = "SimCraft";
  bool vsync = true;
  bool headless = false; // true = 不开窗口，仅离屏渲染/导出帧
};

/// 渲染器接口
class Renderer {
public:
  virtual ~Renderer() = default;

  /// 获取队列引用 — 模拟线程通过此提交帧
  [[nodiscard]] FrameQueue& queue() { return m_queue; }

  /// 在当前线程运行渲染循环（阻塞）。
  /// 通常在主线程调用（满足 GLFW/macOS 窗口事件必须主线程的要求）。
  void runOnCurrentThread() {
    m_running = true;
    renderLoop();
  }

  /// 请求关闭
  void shutdown() {
    m_running = false;
    m_queue.shutdown();
  }

  [[nodiscard]] bool isRunning() const { return m_running; }

  /// 设置用户交互回调（相机操作等，在渲染线程中触发）
  void setInputCallback(std::function<void(CameraState&)> cb) { m_inputCallback = std::move(cb); }

protected:
  /// 子类实现：一次性初始化 (在渲染线程上调用)
  virtual void initialize(const RendererConfig& config) = 0;
  /// 子类实现：绘制一帧
  virtual void drawFrame(const SceneProxy& scene) = 0;
  /// 子类实现：清理资源
  virtual void cleanup() = 0;
  /// 子类实现：处理窗口事件 + swap buffers，返回 false 表示窗口关闭
  virtual bool pollAndSwap() = 0;

  RendererConfig m_config;
  CameraState m_camera;
  std::function<void(CameraState&)> m_inputCallback;

private:
  void renderLoop() {
    initialize(m_config);

    while (m_running) {
      // 1. 阻塞等待下一帧（模拟线程 push 后唤醒）
      auto frame = m_queue.pop();
      if (!frame) break; // shutdown

      // 2. 处理用户输入（可能修改 m_camera）
      if (m_inputCallback)
        m_inputCallback(m_camera);

      // 3. 用当前相机渲染
      frame->camera = m_camera;
      drawFrame(*frame);

      // 4. swap + poll events
      if (!pollAndSwap()) {
        m_running = false;
        m_queue.shutdown(); // 通知模拟线程也别等了
      }
    }

    cleanup();
  }

  FrameQueue m_queue;
  std::atomic<bool> m_running{false};
};

/// 工厂函数
std::unique_ptr<Renderer> createRenderer(const RendererConfig& config = {});

} // namespace sim::renderer
