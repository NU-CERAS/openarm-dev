"""Click drum buttons to execute live arm transfers and J7-only strikes."""
from openarm_drummer.live_demo import LiveDemo, DemoControls, DemoPanel
from openarm_drummer.scene import build_scene
from openarm_drummer.teach import native_app, run_native_window


def main():
    native_app('Live Demo')
    model,robot,scene=build_scene()
    session=LiveDemo(model,robot,scene)
    run_native_window(model,session,DemoPanel(session),DemoControls(session),'OpenDrummer | Live drum demo')


if __name__=='__main__':main()
