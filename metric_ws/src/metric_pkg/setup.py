from setuptools import find_packages, setup

package_name = 'metric_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/tasks.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manasa',
    maintainer_email='gyaramanasa01@gmail.com',
    description='Proving Metrics',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'joint_space=metric_pkg.joint:main',
            'scene_builder=metric_pkg.scene:main',
            'analyze_script=metric_pkg.analyze:main',
            'pose_constraint=metric_pkg.pose:main',
            'pose_generator=metric_pkg.pose_generator_6dof:main',
            'pose_generator_5dof=metric_pkg.pose_generator_5dof:main',
            'poses=metric_pkg.poses_generate_6dof:main',
            'pose_modified=metric_pkg.metrics:main',
            'efficiency=metric_pkg.metrics_for_efficiency:main',
            'benchmark_node=metric_pkg.node:main',
            'results=metric_pkg.results:main',
            'pose_limits=metric_pkg.pose_limits:main',
            'demo2=metric_pkg.demo2:main',
            'pose_generator_new=metric_pkg.poses_generator_new:main',
            'new=metric_pkg.new:main',


        ],
    },
)
