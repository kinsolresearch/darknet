#include "darknet.h"

void train_detector_partial(network *net, int n_batches, char *train_images)
{    
    srand(time(0));
    int imgs = net->batch * net->subdivisions;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;

    double time;
    int count = 0;
    long cur_batch = 0;
    float avg_loss = -1;
    pthread_t load_thread = load_data(args);
    int start_batch = get_current_batch(net);
    while((cur_batch = get_current_batch(net)) < start_batch + n_batches){
        printf("%d / %d\n", cur_batch, start_batch + n_batches);
        if(l.random && count++%10 == 0){ // Not sure exactly what this block does!
            printf("Resizing\n");
            args.w = net->w;
            args.h = net->h;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
        loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf(
            "%ld: %f, %f avg, %f rate, %lf seconds, %ld images\n",
            cur_batch + 1, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, cur_batch*imgs
        );
        free_data(train);

    }
}
