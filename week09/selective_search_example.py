import argparse
import cv2


def main(
    img_path: str,
    method: str = "f",
    img_size: int = 500,
    init_proposals_num: int = 100,
    proposals_increment: int = 50,
) -> None:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    im = cv2.imread(img_path)
    height_new = img_size
    width_new = int(im.shape[1] * img_size / im.shape[0])
    im = cv2.resize(im, (width_new, height_new))

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)

    if method == "f":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()
    print(f"Количество регионов: {len(rects)}")

    while True:
        img_out = im.copy()

        for i, rect in enumerate(rects):
            if i > init_proposals_num:
                break

            x, y, w, h = rect
            cv2.rectangle(
                img=img_out,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        cv2.imshow("Output", img_out)
        k = cv2.waitKey(0) & 0xFF

        if k == 109:
            # press "m"
            init_proposals_num += proposals_increment
        elif k == 108 and init_proposals_num > proposals_increment:
            # press "l"
            init_proposals_num -= proposals_increment
        elif k == 113:
            # press "q"
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str, default="test.png")
    parser.add_argument("--method", type=str, default="f")
    args = parser.parse_args()

    main(img_path=args.img_path, method=args.method)
