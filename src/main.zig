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
            return .{
                .dataset = dataset,
                .random = random,
            };
        }

        pub fn randomize(self: *@This()) void {
            self.random.bytes(std.mem.sliceAsBytes(&self.weights));
        }

        pub fn ones(self: *@This()) void {
            for (0..self.weights.len) |i| {
                self.weights[i] = @as(dtype, 0);
            }
        }

        pub fn zeros(self: *@This()) void {
            for (0..self.weights.len) |i| {
                self.weights[i] = @as(dtype, 1);
            }
        }

        pub fn fullF(self: *@This(), x: dtype) dtype {
            var sum: dtype = @as(dtype, 0);
            inline for (self.weights, 0..length) |w, i| {
                sum += w * funcs[i](x);
            }
            return sum;
        }

        fn fullFVec(self: *@This(), result: []dtype) void {
            for (self.dataset.x, 0..) |x, i| {
                result[i] = self.fullF(x);
            }
        }

        fn residualI(self: *@This(), i: usize) dtype {
            const y = self.dataset.y[i];
            const x = self.dataset.x[i];
            return y - self.fullF(x);
        }

        pub fn residual(self: *@This(), result: []dtype) void {
            for (0..self.dataset.x.len) |i| {
                result[i] = self.residualI(i);
            }
        }

        fn castToFloat(value: anytype) f64 {
            return switch (@typeInfo(@TypeOf(value))) {
                .int => @floatFromInt(value),
                .comptime_int => @floatFromInt(value),
                else => @as(f64, value),
            };
        }

        pub fn mean_square_error(self: *@This()) f64 {
            var sum: dtype = @as(dtype, 0);
            for (0..self.dataset.x.len) |i| {
                const r = self.residualI(i);
                sum += r * r;
            }
            return sum / castToFloat(self.dataset.x.len);
        }

        fn directionI(self: *@This(), comptime i: usize) dtype {
            var sum: dtype = @as(dtype, 0);
            for (0..self.dataset.x.len) |j| {
                const x = self.dataset.x[j];
                const residual_j = self.residualI(j);
                sum += residual_j * funcs[i](x);
            }
            return sum;
        }

        pub fn direction(self: *@This(), result: []dtype) void {
            inline for (0..length) |i| {
                result[i] = self.directionI(i);
            }
        }

        fn directionSquared(self: *@This()) dtype {
            var sum: dtype = @as(dtype, 0);
            inline for (0..length) |i| {
                const direction_i = self.directionI(i);
                sum += direction_i * direction_i;
            }
            return sum;
        }

        /// gradient pi = 2 A^t ( A v - Y )
        /// v^(k+1) = v^k - alpha gradient pi
        /// v^(k+1) = v^k - alpha A^t (A v - Y)
        /// v^(k+1) = v^k + alpha A^t r
        /// v^(k+1) = v^k + alpha d
        pub fn update_weights(self: *@This()) void {
            var direction_vec: [length]dtype = undefined;
            self.direction(&direction_vec);

            for (0..direction_vec.len) |i| {
                self.weights[i] = self.weights[i] + self.learning_rate * direction_vec[i];
            }
        }

        fn qI(self: *@This(), i: usize) dtype {
            var sum: dtype = @as(dtype, 0);
            inline for (0..length) |j| {
                const x = self.dataset.x[i];
                sum += self.directionI(j) * funcs[j](x);
            }
            return sum;
        }

        fn sI(self: *@This(), comptime i: usize) dtype {
            var sum: dtype = @as(dtype, 0);
            for (self.dataset.x, 0..) |x, j| {
                sum += self.qI(j) * funcs[i](x);
            }
            return sum;
        }

        pub fn update_learning_rate(self: *@This()) void {
            const denominator = denominator: {
                var sum: dtype = @as(dtype, 0);
                inline for (0..length) |i| {
                    sum += self.sI(i) * self.directionI(i);
                }
                break :denominator sum;
            };
            self.learning_rate = self.directionSquared() / denominator;
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
    const dtype = f64;
    const funcs = .{
        (struct {
            fn func(x: dtype) dtype {
                return x;
            }
        }).func,
        (struct {
            fn func(x: dtype) dtype {
                _ = x;
                return 1.0;
            }
        }).func,
            // (struct {
            //     fn func(x: dtype) dtype {
            //         return 1 / x;
            //     }
            // }).func,
            // (struct {
            //     fn func(x: dtype) dtype {
            //         return std.math.exp(x);
            //     }
            // }).func,
            // (struct {
            //     fn func(x: dtype) dtype {
            //         return std.math.sin(x);
            //     }
            // }).func,
    };
    const dataset = Dataset(dtype).init(&.{ 1.0, 2.0, 3.0 }, &.{ 4.0, 5.0, 6.0 });
    var gradient_descent = GradientDescent(dtype, funcs).init(dataset, prng.random());
    gradient_descent.learning_rate = 0.01;
    // gradient_descent.randomize();
    gradient_descent.ones();
    gradient_descent.update_learning_rate();
    gradient_descent.update_weights();
    std.debug.print("mean square error: {}\n", .{gradient_descent.mean_square_error()});
    for (0..50) |_| {
        gradient_descent.update_learning_rate();
        gradient_descent.update_weights();
        std.debug.print("mean square error: {}\n", .{gradient_descent.mean_square_error()});
    }
}
