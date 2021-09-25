import numpy as np
import rosbag
import yaml
from ouster import client
import matplotlib.pyplot as plt


class RosbagToPCLExtractor:

    def __init__(self, rosbag_file, topic, metadata_path):
        self.rosbag_file = rosbag_file
        self.topic = topic
        print("Loading rosbag " + self.rosbag_file + "...")
        self.bag = rosbag.Bag(self.rosbag_file)
        print("...done.")

        # Print information and check rosbag -----
        self.num_samples = 0
        info_dict = yaml.load(self.bag._get_yaml_info())
        print("Duration of the bag: " + str(info_dict["duration"]))
        for topic_messages in info_dict["topics"]:
            if topic_messages["topic"] == self.topic:
                self.num_samples = topic_messages["messages"]
        if self.num_samples > 0:
            print("Number of messages for topic " + self.topic + ": " + str(self.num_samples))
        else:
            raise Exception("Topic " + self.topic + " is not present in the given rosbag (" + self.rosbag_file + ").")
        # -----------------------------------------

        with open(metadata_path, 'r') as f:
            self.metadata = client.SensorInfo(f.read())

        self.ls = client.LidarScan(128, 1024)
        self.xyzlut = client.XYZLut(self.metadata)
        self.lidarpkt = None

        self.range_field_a = []
        self.signal_field_a = []
        self.near_ir_field_a = []
        self.reflectivity_field_a = []
        self.meas_id_a = []

        self.kept_frame_id = None
        self.current_frame_id = None

    def _reset(self):
        self.range_field_a = []
        self.signal_field_a = []
        self.near_ir_field_a = []
        self.reflectivity_field_a = []
        self.meas_id_a = []
        self.kept_frame_id = self.current_frame_id

    def _append_lidarpkt(self):
        self.range_field_a.append(self.lidarpkt.field(client.ChanField.RANGE))
        self.signal_field_a.append(self.lidarpkt.field(client.ChanField.SIGNAL))
        self.near_ir_field_a.append(self.lidarpkt.field(client.ChanField.NEAR_IR))
        self.reflectivity_field_a.append(self.lidarpkt.field(client.ChanField.REFLECTIVITY))
        self.meas_id_a.append(self.lidarpkt.header(client.ColHeader.MEASUREMENT_ID))

    def _get_XYZ_for_1frame(self):
        self.range_field_a = np.concatenate(self.range_field_a, axis=1)
        self.signal_field_a = np.concatenate(self.signal_field_a, axis=1)
        self.near_ir_field_a = np.concatenate(self.near_ir_field_a, axis=1)
        self.reflectivity_field_a = np.concatenate(self.reflectivity_field_a, axis=1)
        self.meas_id_a = np.concatenate(self.meas_id_a).tolist()
        if self.range_field_a.shape[1] != 1024:
            range_field = np.zeros((128, 1024))
            range_field[:, self.meas_id_a] = self.range_field_a
            signal_field = np.zeros((128, 1024))
            signal_field[:, self.meas_id_a] = self.signal_field_a
            near_ir_field = np.zeros((128, 1024))
            near_ir_field[:, self.meas_id_a] = self.near_ir_field_a
            refl_field = np.zeros((128, 1024))
            refl_field[:, self.meas_id_a] = self.reflectivity_field_a
            data = np.concatenate([range_field.reshape(1, -1), signal_field.reshape(1, -1),
                                   near_ir_field.reshape(1, -1), refl_field.reshape(1, -1)], axis=0)

        else:
            data = np.concatenate([self.range_field_a.reshape(1, -1), self.signal_field_a.reshape(1, -1),
                                   self.near_ir_field_a.reshape(1, -1), self.reflectivity_field_a.reshape(1, -1)],
                                  axis=0)

        self.ls._data = data
        xyz = self.xyzlut(self.ls)
        return xyz

    def _viz(self, XYZ):
        [x, y, z] = [c.flatten() for c in np.dsplit(XYZ, 3)]
        ax = plt.axes(projection='3d')
        r = 14
        ax.set_xlim3d([-r, r])
        ax.set_ylim3d([-r, r])
        ax.set_zlim3d([-r / 2, r / 2])
        ax.view_init(azim=60)
        # plt.axis('off')
        z_col = np.minimum(np.absolute(z), 5)
        ax.scatter(x, y, z, c=z_col, s=0.2)
        plt.show()

    def preprocess_rosbag(self, show_frame_id=-1):
        for index, (topic, msg, t) in enumerate(self.bag.read_messages(topics=[self.topic])):
            if not index % 10:
                print("Preprocessing scan " + str(
                    index) + "/" + str(self.num_samples) + " from the point cloud " + self.rosbag_file + ".")

            self.lidarpkt = client.LidarPacket(data=msg.buf, info=self.metadata, timestamp=t)
            self.current_frame_id = self.lidarpkt.header(client.ColHeader.FRAME_ID)[0]

            if self.kept_frame_id is None:
                self.kept_frame_id = self.current_frame_id
            if self.current_frame_id == self.kept_frame_id:
                self._append_lidarpkt()
            else:
                xyz = self._get_XYZ_for_1frame()
                np.save('frames27/{}.npy'.format(self.kept_frame_id), xyz)
                print(self.kept_frame_id)
                if show_frame_id == self.kept_frame_id:
                    self._viz(xyz)

                self._reset()
                self._append_lidarpkt()
        self.bag.close()
