#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace zonetool::t7
{
	namespace converter::iw7
	{
		namespace detail
		{
			// One parallel_for call: its indices go to whichever pool threads join, and to the caller.
			struct parallel_job
			{
				std::uint32_t count = 0;
				std::function<void(std::uint32_t, std::uint32_t)> fn;
				std::atomic<std::uint32_t> next{ 0 }; // next index to hand out
				std::atomic<std::uint32_t> finished{ 0 }; // indices done (or skipped after a failure)
				std::atomic<std::uint32_t> slots{ 0 }; // thread slots handed out
				std::mutex error_mutex;
				std::exception_ptr error;
				std::mutex done_mutex;
				std::condition_variable done;

				// runs indices until none are left; `slot` is unique among the threads in this job
				void work(const std::uint32_t slot)
				{
					for (;;)
					{
						const auto i = this->next++;
						if (i >= this->count)
						{
							return;
						}
						try
						{
							this->fn(i, slot);
						}
						catch (...)
						{
							std::lock_guard _(this->error_mutex);
							if (!this->error)
							{
								this->error = std::current_exception();
							}
							// the indices nobody took yet count as finished
							const auto taken = this->next.exchange(this->count);
							if (taken < this->count)
							{
								this->complete(this->count - taken);
							}
						}
						this->complete(1);
					}
				}

				void complete(const std::uint32_t n)
				{
					if (this->finished.fetch_add(n) + n == this->count)
					{
						std::lock_guard _(this->done_mutex);
						this->done.notify_all();
					}
				}
			};

			// Threads started once for the whole run: every thread start and exit runs each loaded DLL's attach /
			// detach under the loader lock, which would serialize concurrent callers.
			class thread_pool
			{
			public:
				static thread_pool& get()
				{
					// never destroyed: its detached threads wait on its members until the process ends
					static auto* pool = new thread_pool();
					return *pool;
				}

				std::uint32_t size() const
				{
					return static_cast<std::uint32_t>(this->threads_.size());
				}

				void submit(const std::shared_ptr<parallel_job>& job)
				{
					{
						std::lock_guard _(this->mutex_);
						this->jobs_.push_back(job);
					}
					this->wake_.notify_all();
				}

			private:
				thread_pool()
				{
					const auto count = std::max(1u, std::thread::hardware_concurrency());
					for (auto i = 0u; i < count; i++)
					{
						this->threads_.emplace_back([this]
						{
							this->loop();
						});
						this->threads_.back().detach();
					}
				}

				void loop()
				{
					for (;;)
					{
						std::shared_ptr<parallel_job> job;
						{
							std::unique_lock lock(this->mutex_);
							this->wake_.wait(lock, [&]
							{
								// jobs with nothing left to hand out leave the queue
								while (!this->jobs_.empty() && this->jobs_.front()->next >= this->jobs_.front()->count)
								{
									this->jobs_.pop_front();
								}
								return !this->jobs_.empty();
							});
							job = this->jobs_.front();
						}
						job->work(job->slots++);
					}
				}

				std::mutex mutex_;
				std::condition_variable wake_;
				std::deque<std::shared_ptr<parallel_job>> jobs_;
				std::vector<std::thread> threads_;
			};
		}

		// fn(i, thread) for every i in [0, count), on the shared pool's threads and the caller's; `thread` is below
		// std::thread::hardware_concurrency() + 1 and unique among the threads running this call. The first exception
		// thrown by any call stops the rest and is rethrown here.
		template <typename F>
		void parallel_for(const std::uint32_t count, F&& fn)
		{
			if (!count)
			{
				return;
			}
			auto job = std::make_shared<detail::parallel_job>();
			job->count = count;
			job->fn = [&fn](const std::uint32_t i, const std::uint32_t slot)
			{
				fn(i, slot);
			};
			if (count > 1)
			{
				detail::thread_pool::get().submit(job);
			}
			job->work(job->slots++);
			{
				std::unique_lock lock(job->done_mutex);
				job->done.wait(lock, [&]
				{
					return job->finished >= job->count;
				});
			}
			if (job->error)
			{
				std::rethrow_exception(job->error);
			}
		}
	}
}
