import numpy as np
import rosbag
import yaml
from ouster import client
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


class RosbagToIMGExtractor:

    def __init__(self, rosbag_file, topics, metadata_path):
        self.rosbag_file = rosbag_file
        self.topics = topics
        print("Loading rosbag " + self.rosbag_file + "...")
        self.bag = rosbag.Bag(self.rosbag_file)
        print("...done.")

        # Print information and check rosbag -----
        self.num_samples = 0
        info_dict = yaml.load(self.bag._get_yaml_info())
        print("Duration of the bag: " + str(info_dict["duration"]))
        for topic_messages in info_dict["topics"]:
            if topic_messages["topic"] == self.topics[0]:
                self.num_samples = topic_messages["messages"]
        if self.num_samples > 0:
            print("Number of messages for topic " + self.topics[0] + ": " + str(self.num_samples))
        else:
            raise Exception(
                "Topic " + self.topics[0] + " is not present in the given rosbag (" + self.rosbag_file + ").")
        # -----------------------------------------
        with open(metadata_path, 'r') as f:
            self.metadata = client.SensorInfo(f.read())
        self.kept_frame_id = None
        self.current_frame_id = None
        self.kept_img_msg = None

    def preprocess_rosbag(self, save_path):
        bridge = CvBridge()
        for index, (topic, msg, t) in enumerate(self.bag.read_messages(topics=self.topics)):
            if not index % 100:
                print("Preprocessing scan " + str(
                    index) + "/" + str(self.num_samples) + " from the point cloud " + self.rosbag_file + ".")

            if topic == self.topics[1]:
                self.lidarpkt = client.LidarPacket(data=msg.buf, info=self.metadata)
                self.current_frame_id = self.lidarpkt.header(client.ColHeader.FRAME_ID)[0]
                if self.kept_frame_id is None:
                    self.kept_frame_id = self.current_frame_id
            else:
                self.kept_img_msg = msg

            if self.current_frame_id != self.kept_frame_id:
                if self.kept_img_msg is not None:
                    cv_img = bridge.imgmsg_to_cv2(self.kept_img_msg, desired_encoding='rgb8')
                    plt.imsave(save_path + '/image{}.png'.format(self.kept_frame_id), cv_img)

                self.kept_frame_id = self.current_frame_id
                self.kept_img_msg = None

        self.bag.close()


if __name__ == "__main__":
    rosbag_file = 'CamerasAndLidars_2021-08-27-12-59-11.bag'
    metadata_path = 'os-992031000052.local.json'
    topic1 = '/camMainView/Downsampled'
    topic2 = '/os_node/lidar_packets'
    topics = [topic1, topic2]
    save_path = 'cam'

    extractor = RosbagToIMGExtractor(rosbag_file, topics, metadata_path)
    extractor.preprocess_rosbag(save_path)
