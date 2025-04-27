const std = @import("std");

pub fn Dataset(comptime dtype: type) type {
    return struct {
        x: []const dtype,
        y: []const dtype,

        pub fn init(x: []const dtype, y: []const dtype) @This() {
            return .{
                .x = x,
                .y = y,
            };
        }
    };
}

pub fn GradientDescent(comptime dtype: type, comptime funcs: anytype) type {
    const length = funcs.len;
    return struct {
        dataset: Dataset(dtype),
        weights: [length]dtype = .{@as(dtype, 0)} ** length,
        random: std.Random,
        learning_rate: dtype = @as(dtype, 0),
        pub fn init(dataset: Dataset(dtype), random: std.Random) @This() {
            var learning_rate: dtype = undefined;
            random.bytes(std.mem.asBytes(&learning_rate));
            return .{
                .dataset = dataset,
                .random = random,
                .learning_rate = learning_rate,
            };
        }

        pub fn randomize(self: *@This()) void {
            self.random.bytes(std.mem.sliceAsBytes(&self.weights));
        }

        pub fn full_f(weights: []dtype, x: dtype) dtype {
            var sum: dtype = @as(dtype, 0);
            inline for (weights, 0..length) |w, i| {
                sum += w * funcs[i](x);
            }
            return sum;
        }

        pub fn full_f_vec(weights: []dtype, xs: []dtype, result: []dtype) void {
            for (xs, 0..) |x, i| {
                result[i] = full_f(weights, x);
            }
        }

        pub fn residual(dataset: Dataset(dtype), weights: []dtype, result: []dtype) void {
            for (dataset.y, dataset.x, 0..) |x, y, i| {
                result[i] = y - full_f(weights, x);
            }
        }

        fn castToFloat(value: anytype) f64 {
            return switch (@typeInfo(@TypeOf(value))) {
                .int => @floatFromInt(value),
                .comptime_int => @floatFromInt(value),
                else => @as(f64, value),
            };
        }

        pub fn mean_square_error(dataset: Dataset(dtype), weights: []dtype) f64 {
            var sum: dtype = @as(dtype, 0);
            for (dataset.y, dataset.x) |y, x| {
                const r = y - full_f(weights, x);
                sum += r * r;
            }
            return sum / castToFloat(dataset.x.len);
        }

        pub fn direction(dataset: Dataset(dtype), weights: []dtype, result: []dtype) void {
            inline for (0..length) |i| {
                result[i] = @as(dtype, 0);
                for (dataset.y, dataset.x) |y, x| {
                    const residual_j = y - full_f(weights, x);
                    result[i] += residual_j * funcs[i](x);
                }
            }
        }

        /// v^(k+1) = v^k - alpha gradient pi
        pub fn update_weights(self: *@This()) void {
            var direction_vec: [length]dtype = undefined;
            direction(self.dataset, &self.weights, &direction_vec);

            for (0..direction_vec.len) |i| {
                self.weights[i] = self.weights[i] + self.learning_rate * direction_vec[i];
            }
        }
    };
}

pub fn main() !void {
    // Create a random number generator
    var prng = std.Random.DefaultPrng.init(blk: {
        // SAFETY: defined immediatly after
        // var seed: u64 = undefined;
        // std.crypto.random.bytes(std.mem.asBytes(&seed));
        // break :blk seed;
        break :blk 1;
    });
    const funcs = .{
        (struct {
            fn func(x: f32) f32 {
                return x * x;
            }
        }).func,
        (struct {
            fn func(x: f32) f32 {
                return 1 / x;
            }
        }).func,
            // (struct {
            //     fn func(x: f32) f32 {
            //         return std.math.exp(x);
            //     }
            // }).func,
            // (struct {
            //     fn func(x: f32) f32 {
            //         return std.math.sin(x);
            //     }
            // }).func,
    };
    const dataset = Dataset(f32).init(&.{ 1.0, 2.0, 3.0 }, &.{ 4.0, 5.0, 6.0 });
    const GradientDescentType = GradientDescent(f32, funcs);
    var gradient_descent = GradientDescentType.init(dataset, prng.random());
    gradient_descent.learning_rate = 0.01;
    gradient_descent.randomize();
    for (0..50) |_| {
        gradient_descent.update_weights();
        std.debug.print("mean square error: {}\n", .{GradientDescentType.mean_square_error(dataset, &gradient_descent.weights)});
    }
}
